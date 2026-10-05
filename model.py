#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
import torch.nn.init as init

from torch.utils.data import DataLoader
from model_cond import Diffusion_Cond
from dataloader import TestDataset
import time
# replace the whole KGEmodel, add model_cond,model_diffusion

def get_geometric_condition(emb_h, emb_r, model_name):
    """
    Tail prediction: (h, r, ?) -> t
    """
    if model_name == 'TransE':
        return emb_h + emb_r
    if model_name == 'DistMult':
        return emb_h * emb_r
    if model_name == 'RotatE' or model_name == 'ComplEx':
        re_h, im_h = torch.chunk(emb_h, 2, dim=-1)
        re_r, im_r = torch.chunk(emb_r, 2, dim=-1)
        re_t = re_h * re_r - im_h * im_r
        im_t = re_h * im_r + im_h * re_r
        return torch.cat([re_t, im_t], dim=-1)
    return emb_h + emb_r


def get_inverse_geometric_condition(emb_t, emb_r, model_name):
    """
    Head prediction: (?, r, t) -> h
    """
    if model_name == 'TransE':
        return emb_t - emb_r
    if model_name == 'DistMult':
        return emb_t * emb_r
    if model_name == 'RotatE' or model_name == 'ComplEx':
        re_t, im_t = torch.chunk(emb_t, 2, dim=-1)
        re_r, im_r = torch.chunk(emb_r, 2, dim=-1)
        re_h = re_t * re_r + im_t * im_r
        im_h = im_t * re_r - re_t * im_r
        return torch.cat([re_h, im_h], dim=-1)
    return emb_t - emb_r


def get_adversarial_condition(condition_emb, input_emb_list, epsilon=0.01):
    """
    FGSM-style adversarial perturbation on condition embeddings.
    """
    grads = []
    for emb in input_emb_list:
        if emb.grad is not None:
            grads.append(emb.grad)
    if not grads:
        return condition_emb

    total_grad = torch.zeros_like(condition_emb)
    for g in grads:
        if g.shape == condition_emb.shape:
            total_grad += g

    grad_norm = total_grad.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    total_grad = total_grad / grad_norm
    adv_condition = condition_emb + epsilon * total_grad.detach()
    return adv_condition


'''门控控制函数'''
def proto_gating_weight(z_raw, r_ids, proto_table, beta=1.0, tau=0.3, clamp=10.0):
    """
    输入：
        - z_raw：原始嵌入空间中的候选虚拟负样本，形状为 [B, N, D]
        - r_ids：当前 batch 的关系 id，形状为 [B]
        - proto_table：关系原型表，形状为 [nrelation, M, D]
        - beta / tau / clamp：原型门控强度与稳定性超参数
    功能：
        - 根据候选样本与关系原型之间的语义相似度计算门控权重
        - 提升与当前关系语义更一致的虚拟负样本权重
    输出：
        - w：形状为 [B, N] 的门控权重；若未启用原型表则返回 None
    """
    """
    z_raw: [B, N, D] candidate negatives in raw space
    r_ids: [B] relation ids
    proto_table: [nrelation, M, D]
    return: [B, N] gating weights
    """
    if proto_table is None or beta <= 0:
        return None

    # Pick relation-specific prototypes: [B, M, D]
    proto = proto_table[r_ids]

    # Prototype tables are usually normalized offline; normalize here for safety.
    z = F.normalize(z_raw, dim=-1)
    proto = F.normalize(proto, dim=-1)

    sim = torch.einsum('bnd,bmd->bnm', z, proto) / max(float(tau), 1e-6)
    g = torch.logsumexp(sim, dim=-1)
    w = torch.exp(float(beta) * g).clamp_max(float(clamp))
    return w

'''多种子互斥轨迹的排斥损失 约束不同 seed 的扩散轨迹方向不要过于相似缓解多轨迹采样塌缩到少数方向的问题'''
def seed_repulsion_loss(delta_bskd, kept_idx, timesteps, tau=0.3, t_min=0.3, t_max=0.8, eps=1e-12):
    B, S, K, D = delta_bskd.shape
    if S <= 1 or K == 0:
        zero = delta_bskd.new_tensor(0.0)
        return zero, zero

    if kept_idx is None:
        t_mask = delta_bskd.new_ones((K,), dtype=torch.bool)
    else:
        if not torch.is_tensor(kept_idx):
            kept_idx = torch.as_tensor(kept_idx, device=delta_bskd.device)
        else:
            kept_idx = kept_idx.to(delta_bskd.device)
        t_norm = kept_idx.float() / float(max(timesteps - 1, 1))
        t_mask = (t_norm >= float(t_min)) & (t_norm <= float(t_max))

    if not t_mask.any():
        zero = delta_bskd.new_tensor(0.0)
        return zero, zero

    delta = delta_bskd[:, :, t_mask, :]  # [B,S,K',D]
    B, S, Kp, D = delta.shape
    delta = delta / delta.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

    x = delta.permute(0, 2, 1, 3).contiguous().view(B * Kp, S, D)  # [B*K',S,D]
    G = torch.bmm(x, x.transpose(1, 2))  # [B*K',S,S]

    eye = torch.eye(S, device=delta_bskd.device, dtype=G.dtype).unsqueeze(0)
    off_mask = (1.0 - eye)
    off = G * off_mask

    loss = torch.exp(off / max(float(tau), 1e-6)).mean()

    off_sum = off.sum(dim=(1, 2))
    off_cnt = off_mask.sum(dim=(1, 2)).clamp_min(1.0)
    off_mean = (off_sum / off_cnt).mean()
    return loss, off_mean

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False, score_scale=1.0):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.double_entity_embedding = double_entity_embedding
        self.double_relation_embedding = double_relation_embedding
        self.score_scale = float(score_scale)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        diff_dim = 64 if double_entity_embedding else 32

        self.linear_layer = nn.Linear(self.entity_dim, diff_dim)
        self.linear_layer_diff = nn.Linear(diff_dim, self.entity_dim)
        self.linear_layer_relation = nn.Linear(self.relation_dim, diff_dim)
        self.linear_layer_dim = nn.Linear(64, 32) # if de dr ,用来将关系转换成 32
        #
        self._initialize_weights()

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # undirected graph : nrelation*2
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def _initialize_weights(self):
        init.xavier_uniform_(self.linear_layer.weight)
        init.xavier_uniform_(self.linear_layer_diff.weight)
        init.xavier_uniform_(self.linear_layer_relation.weight)
        init.xavier_uniform_(self.linear_layer_dim.weight)



    '''根据三元组打分传递梯度到主模型'''

    def forward(self, sample, mode='single', method=None, h_emb_diff=None, t_emb_diff= None, h_emb=None, r_emb=None, t_emb=None):
        if method == 'id':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:,0]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=sample[:,1]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:,2]
                ).unsqueeze(1)

            elif mode == 'head-batch':
                tail_part, head_part = sample
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

                head = torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=head_part.view(-1)
                        ).view(batch_size, negative_sample_size, -1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)


                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            else:
                raise ValueError('mode %s not supported' % mode)
        else:
            if mode == 'single':
                if h_emb is not None and r_emb is not None and t_emb is not None:
                    head = h_emb
                    relation = r_emb
                    tail = t_emb
                else:
                    batch_size, negative_sample_size = sample.size(0), 1
                    head = torch.index_select(
                        self.entity_embedding,
                        dim=0,
                        index=sample[:,0]
                    ).unsqueeze(1)

                    relation = torch.index_select(
                        self.relation_embedding,
                        dim=0,
                        index=sample[:,1]
                    ).unsqueeze(1)

                    tail = torch.index_select(
                        self.entity_embedding,
                        dim=0,
                        index=sample[:,2]
                    ).unsqueeze(1)

            elif mode == 'head-batch':
                head = h_emb_diff if h_emb_diff is not None else h_emb
                relation = r_emb
                tail = t_emb

            elif mode == 'tail-batch':
                head = h_emb
                relation = r_emb
                tail = t_emb_diff if t_emb_diff is not None else t_emb




            else:
                raise ValueError('mode %s not supported' % mode)


        # if head is not None and head.shape[2] == 1000:
        #     head = self.linear_layer(head)
        # if relation is not None and relation.shape[2] == 1000:
        #     relation = self.linear_layer(relation)
        # if tail is not None and tail.shape[2] == 1000:
        #     tail = self.linear_layer(tail)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2) * self.score_scale
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2) * self.score_scale
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        # score = score.norm(dim=0)
        score = torch.sqrt(score.pow(2).sum(dim=0) + 1e-9)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:

            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, diffusion_head, d_optimizer_head, diffusion_tail, d_optimizer_tail, neighbors, relations):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        num = 20

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        h_emb_all = model.linear_layer(model.entity_embedding)
        r_emb_all = model.linear_layer_relation(model.relation_embedding)

        head_ids = positive_sample[:, 0]
        rel_ids = positive_sample[:, 1]
        tail_ids = positive_sample[:, 2]

        h_emb = h_emb_all.index_select(0, head_ids)
        r_emb = r_emb_all.index_select(0, rel_ids)
        t_emb = h_emb_all.index_select(0, tail_ids)

        h_raw = model.entity_embedding.index_select(0, head_ids)
        r_raw = model.relation_embedding.index_select(0, rel_ids)
        t_raw = model.entity_embedding.index_select(0, tail_ids)

        output_detached = h_emb_all.detach()
        output_relation_detached = r_emb_all.detach()

        def raw_query_tail_condition(h_raw_vec, r_raw_vec):
            # Build condition in raw KGE space, then project to diffusion space later.
            if args.model == 'TransE':
                return h_raw_vec + r_raw_vec
            if args.model == 'DistMult':
                return h_raw_vec * r_raw_vec
            if args.model == 'ComplEx':
                re_h, im_h = torch.chunk(h_raw_vec, 2, dim=-1)
                re_r, im_r = torch.chunk(r_raw_vec, 2, dim=-1)
                re_t = re_h * re_r - im_h * im_r
                im_t = re_h * im_r + im_h * re_r
                return torch.cat([re_t, im_t], dim=-1)
            if args.model == 'RotatE':
                pi = 3.14159265358979323846
                re_h, im_h = torch.chunk(h_raw_vec, 2, dim=-1)
                phase_r = r_raw_vec / (model.embedding_range.item() / pi)
                re_r = torch.cos(phase_r)
                im_r = torch.sin(phase_r)
                re_t = re_h * re_r - im_h * im_r
                im_t = re_h * im_r + im_h * re_r
                return torch.cat([re_t, im_t], dim=-1)
            return h_raw_vec + r_raw_vec

        def raw_query_head_condition(t_raw_vec, r_raw_vec):
            # Build inverse-query condition in raw KGE space.
            if args.model == 'TransE':
                return t_raw_vec - r_raw_vec
            if args.model == 'DistMult':
                return t_raw_vec * r_raw_vec
            if args.model == 'ComplEx':
                re_t, im_t = torch.chunk(t_raw_vec, 2, dim=-1)
                re_r, im_r = torch.chunk(r_raw_vec, 2, dim=-1)
                re_h = re_t * re_r + im_t * im_r
                im_h = im_t * re_r - re_t * im_r
                return torch.cat([re_h, im_h], dim=-1)
            if args.model == 'RotatE':
                pi = 3.14159265358979323846
                re_t, im_t = torch.chunk(t_raw_vec, 2, dim=-1)
                phase_r = r_raw_vec / (model.embedding_range.item() / pi)
                re_r = torch.cos(phase_r)
                im_r = torch.sin(phase_r)
                re_h = re_t * re_r + im_t * im_r
                im_h = im_t * re_r - re_t * im_r
                return torch.cat([re_h, im_h], dim=-1)
            return t_raw_vec - r_raw_vec

        # Positive score (raw) for delta computation and loss; compute once for reuse.
        positive_score_raw = model(positive_sample, method='id').squeeze(dim=1)

        # Virtual branch enable switch: only build diffusion when loss weight >0 and we actually sample something.
        virt_weight_cfg = float(getattr(args, 'virt_loss_weight', 0.05))
        num_seeds_cfg = int(getattr(args, 'num_seeds', 5))
        enable_virtual = virt_weight_cfg > 0.0 and (num_seeds_cfg > 0)

        def train_diffusion():
            # Train conditional diffusion on the current batch (no neighbor index is used)
            h_batch = h_emb_all.index_select(0, head_ids).detach()
            t_batch = h_emb_all.index_select(0, tail_ids).detach()
            h_raw_batch = h_raw.detach()
            r_raw_batch = r_raw.detach()
            t_raw_batch = t_raw.detach()
            total_loss = 0.0
            total_count = 0

            if mode == 'head-batch':
                cond_raw = raw_query_head_condition(t_raw_batch, r_raw_batch)
                cond = model.linear_layer(cond_raw).detach()
                for _ in range(args.d_epoch):
                    d_optimizer_head.zero_grad()
                    '''头尾扩散损失'''
                    dif_loss_head = diffusion_head(h_batch, cond)
                    total_loss += float(dif_loss_head.item())
                    total_count += 1
                    dif_loss_head.backward()
                    d_optimizer_head.step()
            elif mode == 'tail-batch':
                cond_raw = raw_query_tail_condition(h_raw_batch, r_raw_batch)
                cond = model.linear_layer(cond_raw).detach()
                for _ in range(args.d_epoch):
                    d_optimizer_tail.zero_grad()
                    dif_loss_tail = diffusion_tail(t_batch, cond)
                    total_loss += float(dif_loss_tail.item())
                    total_count += 1
                    dif_loss_tail.backward()
                    d_optimizer_tail.step()
            if total_count == 0:
                return None
            return total_loss / float(total_count)
        cur_step = int(getattr(args, 'current_step', 0))
        stage2_start = int(getattr(args, 'stage2_start_step', 0))
        rel_step = max(0, cur_step - stage2_start)
        diff_warm = int(getattr(args, 'diff_warmup_steps', 20000))
        diff_train_loss = None
        if enable_virtual and rel_step >= diff_warm and (mode == 'tail-batch' or mode == 'head-batch'):
            diff_train_loss = train_diffusion()

        
        # ---------------- Mix-and-Diffuse virtual hard negatives (Innovation #1) ----------------
        # Hyper-params
        num_seeds = num_seeds_cfg
        epsilon = float(getattr(args, 'adv_epsilon', 0.005))

        # Mix removed: diffusion-only virtual negatives
        keep_start = int(getattr(args, 'diff_keep_start', 6))
        keep_end = int(getattr(args, 'diff_keep_end', 15))
        init_sigma = float(getattr(args, 'diff_init_sigma', 1.0))
        if args.timesteps is not None:
            keep_start = max(0, min(keep_start, args.timesteps - 1))
            keep_end = max(0, min(keep_end, args.timesteps - 1))
            if keep_end < keep_start:
                keep_start, keep_end = 0, min(9, args.timesteps - 1)
        if bool(getattr(args, 'diff_keep_full', False)) and args.timesteps is not None:
            keep_start = 0
            keep_end = max(0, args.timesteps - 1)

        # Helper: score candidates in raw embedding space (supports (B,N,D))
        def score_candidates_tail(h_raw_vec, r_raw_vec, cand_tail_raw):
            B, N, _ = cand_tail_raw.shape
            return model(
                None, mode='single', method='embedding',
                h_emb=h_raw_vec.unsqueeze(1).expand(B, N, -1),
                r_emb=r_raw_vec.unsqueeze(1).expand(B, N, -1),
                t_emb=cand_tail_raw
            )

        def score_candidates_head(cand_head_raw, r_raw_vec, t_raw_vec):
            B, N, _ = cand_head_raw.shape
            return model(
                None, mode='single', method='embedding',
                h_emb=cand_head_raw,
                r_emb=r_raw_vec.unsqueeze(1).expand(B, N, -1),
                t_emb=t_raw_vec.unsqueeze(1).expand(B, N, -1)
            )


        negative_score_diff = None
        diff_logits = None
        diff_time_weight = None
        diff_raw = None
        rep_loss = None
        rep_offdiag_cos = 0.0
        rep_d_optimizer = None
        virt_detach_query = bool(getattr(args, 'virt_detach_query', True))

        if enable_virtual and rel_step >= diff_warm and (mode == 'tail-batch' or mode == 'head-batch'):
            if mode == 'tail-batch':
                query_cond_raw = raw_query_tail_condition(h_raw.detach(), r_raw.detach())
                anchor_alpha = float(getattr(args, 'anchor_cond_alpha', 0.3))
                anchor_alpha = max(0.0, min(1.0, anchor_alpha))
                base_raw = (1.0 - anchor_alpha) * query_cond_raw + anchor_alpha * t_raw.detach()
                z0 = model.linear_layer(base_raw).detach()  # (B, d)
                cond = z0

                samples, kept_idx = diffusion_tail.sample_multi_seed_from_anchor(
                    (z0.size(0), z0.size(1)),
                    cond,
                    z0,
                    num_seeds=num_seeds,
                    keep_start=keep_start,
                    keep_end=keep_end,
                    init_sigma=init_sigma
                )
                B, S, Kkeep, d = samples.shape
                if float(getattr(args, 'div_weight', 0.0)) > 0:
                    delta_bskd = samples - z0.view(B, 1, 1, d)
                    rep_loss, rep_offdiag = seed_repulsion_loss(
                        delta_bskd=delta_bskd,
                        kept_idx=kept_idx,
                        timesteps=int(args.timesteps),
                        tau=float(getattr(args, 'div_tau', 0.3)),
                        t_min=float(getattr(args, 'div_t_min', 0.3)),
                        t_max=float(getattr(args, 'div_t_max', 0.8))
                    )
                    rep_offdiag_cos = float(rep_offdiag.detach().item())
                    rep_d_optimizer = d_optimizer_tail
                if kept_idx is not None:
                    if not torch.is_tensor(kept_idx):
                        kept_idx = torch.as_tensor(kept_idx, device=z0.device)
                    else:
                        kept_idx = kept_idx.to(z0.device)
                    t_norm = kept_idx.float() / float(max(args.timesteps - 1, 1))
                    mu_t = float(getattr(args, 'virt_t_mu', 0.5))
                    sigma_t = float(getattr(args, 'virt_t_sigma', 0.2))
                    w_t = torch.exp(-0.5 * ((t_norm - mu_t) / max(sigma_t, 1e-6)) ** 2)
                    w_t = w_t.repeat(S)  # (S*Kkeep,)
                    diff_time_weight = w_t.unsqueeze(0).expand(B, -1)
                delta = (samples - z0.view(B, 1, 1, d)).reshape(-1, d)
                delta_scale = float(getattr(args, 'diff_delta_scale', 1.0))
                delta_raw = (delta_scale * model.linear_layer_diff(delta)).view(B, S * Kkeep, -1)
                diff_raw = base_raw.unsqueeze(1) + delta_raw
                emb_range = float(model.embedding_range.item())
                # clamp is safer for TransE/DistMult; for RotatE/ComplEx hard clipping can create boundary artifacts.
                if args.model in ['TransE', 'DistMult']:
                    diff_raw = torch.clamp(diff_raw, -emb_range, emb_range)
                else:
                    diff_raw = torch.nan_to_num(diff_raw, nan=0.0, posinf=emb_range, neginf=-emb_range)
                    diff_raw = emb_range * torch.tanh(diff_raw / emb_range)

                h_for_diff = h_raw.detach() if virt_detach_query else h_raw
                r_for_diff = r_raw.detach() if virt_detach_query else r_raw
                diff_logits = score_candidates_tail(h_for_diff, r_for_diff, diff_raw)  # (B, S*Kkeep)
                # Sanitize diff_logits to avoid NaNs/Infs nuking the whole batch
                diff_logits = torch.nan_to_num(diff_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                nan_mask = torch.isnan(diff_logits)
                if nan_mask.any():
                    diff_logits = diff_logits.masked_fill(nan_mask, -1e9)
                negative_score_diff = diff_logits

            else:  # head-batch
                query_cond_raw = raw_query_head_condition(t_raw.detach(), r_raw.detach())
                anchor_alpha = float(getattr(args, 'anchor_cond_alpha', 0.3))
                anchor_alpha = max(0.0, min(1.0, anchor_alpha))
                base_raw = (1.0 - anchor_alpha) * query_cond_raw + anchor_alpha * h_raw.detach()
                z0 = model.linear_layer(base_raw).detach()
                cond = z0

                samples, kept_idx = diffusion_head.sample_multi_seed_from_anchor(
                    (z0.size(0), z0.size(1)),
                    cond,
                    z0,
                    num_seeds=num_seeds,
                    keep_start=keep_start,
                    keep_end=keep_end,
                    init_sigma=init_sigma
                )
                B, S, Kkeep, d = samples.shape
                if float(getattr(args, 'div_weight', 0.0)) > 0:
                    delta_bskd = samples - z0.view(B, 1, 1, d)
                    rep_loss, rep_offdiag = seed_repulsion_loss(
                        delta_bskd=delta_bskd,
                        kept_idx=kept_idx,
                        timesteps=int(args.timesteps),
                        tau=float(getattr(args, 'div_tau', 0.3)),
                        t_min=float(getattr(args, 'div_t_min', 0.3)),
                        t_max=float(getattr(args, 'div_t_max', 0.8))
                    )
                    rep_offdiag_cos = float(rep_offdiag.detach().item())
                    rep_d_optimizer = d_optimizer_head
                if kept_idx is not None:
                    if not torch.is_tensor(kept_idx):
                        kept_idx = torch.as_tensor(kept_idx, device=z0.device)
                    else:
                        kept_idx = kept_idx.to(z0.device)
                    t_norm = kept_idx.float() / float(max(args.timesteps - 1, 1))
                    mu_t = float(getattr(args, 'virt_t_mu', 0.5))
                    sigma_t = float(getattr(args, 'virt_t_sigma', 0.2))
                    w_t = torch.exp(-0.5 * ((t_norm - mu_t) / max(sigma_t, 1e-6)) ** 2)
                    w_t = w_t.repeat(S)  # (S*Kkeep,)
                    diff_time_weight = w_t.unsqueeze(0).expand(B, -1)
                delta = (samples - z0.view(B, 1, 1, d)).reshape(-1, d)
                delta_scale = float(getattr(args, 'diff_delta_scale', 1.0))
                delta_raw = (delta_scale * model.linear_layer_diff(delta)).view(B, S * Kkeep, -1)
                diff_raw = base_raw.unsqueeze(1) + delta_raw
                emb_range = float(model.embedding_range.item())
                # clamp is safer for TransE/DistMult; for RotatE/ComplEx hard clipping can create boundary artifacts.
                if args.model in ['TransE', 'DistMult']:
                    diff_raw = torch.clamp(diff_raw, -emb_range, emb_range)
                else:
                    diff_raw = torch.nan_to_num(diff_raw, nan=0.0, posinf=emb_range, neginf=-emb_range)
                    diff_raw = emb_range * torch.tanh(diff_raw / emb_range)

                r_for_diff = r_raw.detach() if virt_detach_query else r_raw
                t_for_diff = t_raw.detach() if virt_detach_query else t_raw
                diff_logits = score_candidates_head(diff_raw, r_for_diff, t_for_diff)  # (B, S*Kkeep)
                # Sanitize diff_logits to avoid NaNs/Infs nuking the whole batch
                diff_logits = torch.nan_to_num(diff_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                nan_mask = torch.isnan(diff_logits)
                if nan_mask.any():
                    diff_logits = diff_logits.masked_fill(nan_mask, -1e9)
                negative_score_diff = diff_logits

        random_neg_score = model((positive_sample, negative_sample), mode=mode, method='id')

        # (A) random negatives: keep self-adv if enabled
        adv_weight_rand = 1.0
        adv_weight_virtual = 0.0
        adv_top1_virtual = 0.0
        if args.negative_adversarial_sampling:
            adv_weights = F.softmax(random_neg_score * args.adversarial_temperature, dim=1).detach()
            negative_score = (adv_weights * F.logsigmoid(-random_neg_score)).sum(dim=1)
            adv_weight_rand = 1.0
        else:
            negative_score = F.logsigmoid(-random_neg_score).mean(dim=1)

        # (B) virtual negatives: denoise/robust weighting (diff only)
        virt_loss = None
        virt_weight = 0.0
        diff_weight_mean = 0.0
        w_safe_mean = 0.0
        w_diff_den_mean = 0.0
        frac_den_small = 0.0
        virt_band_coverage = 0.0
        virt_low_cov_skip = 0.0
        num_diff_used = 0.0
        diff_mean_logit = 0.0
        diff_std_logit = 0.0
        diff_max_logit = 0.0
        band_mask = None
        if negative_score_diff is not None:
            # hyperparams
            beta = float(getattr(args, 'virt_beta', 2.0))
            delta_thr = float(getattr(args, 'virt_delta', 0.0))
            band_max = float(getattr(args, 'virt_band_max', 2.0))
            min_cov = float(getattr(args, 'virt_min_coverage', 0.01))
            mu_n = float(getattr(args, 'virt_mu_n', getattr(args, 'virt_mu', 1.0)))
            sigma_n = float(getattr(args, 'virt_sigma_n', getattr(args, 'virt_sigma', 0.5)))

            pos_for_w = positive_score_raw.detach().unsqueeze(1)
            # diff weights
            diff_loss = None
            if diff_logits is not None and diff_logits.numel() > 0:
                delta_diff = pos_for_w - diff_logits
                delta_rand = pos_for_w - random_neg_score.detach()
                delta_std = delta_rand.std(dim=1, keepdim=True).clamp_min(1e-6)
                delta_n = delta_diff / delta_std

                # Allow ultra-hard negatives (delta_n <= 0) to pass band filtering.
                band_mask = (delta_n < band_max)
                virt_band_coverage = band_mask.float().mean().item()
                num_diff_used = band_mask.float().sum(dim=1).mean().item()

                if virt_band_coverage < min_cov:
                    virt_low_cov_skip = 1.0
                else:
                    # 针对ComplEx/DistMult: 过滤过于极端的负样本（可能是假负样本）
                    if args.model in ['ComplEx', 'DistMult']:
                        w_safe_diff = torch.where(
                            delta_n <= -3.0,  # 过于极端的负样本直接丢弃
                            torch.zeros_like(delta_n),
                            torch.where(
                                delta_n <= 0,
                                torch.ones_like(delta_n) * 0.3,  # 降低超难样本权重
                                torch.sigmoid(beta * (delta_n - delta_thr))
                            )
                        ) * band_mask.float()
                    else:
                        # TransE/RotatE: 保持原有逻辑
                        w_safe_diff = torch.where(
                            delta_n <= 0,
                            torch.ones_like(delta_n) * 0.3,
                            torch.sigmoid(beta * (delta_n - delta_thr))
                        ) * band_mask.float()
                    w_safe_mean = w_safe_diff.mean().item()
                    # Keep the sweet-spot weighting active for ultra-hard samples too;
                    # otherwise false negatives can be unintentionally boosted back up.
                    w_delta_diff = torch.exp(
                        -0.5 * ((delta_n - mu_n) / max(sigma_n, 1e-6)) ** 2
                    )
                    if diff_time_weight is not None:
                        w_t = diff_time_weight
                    else:
                        w_t = torch.ones_like(diff_logits)
                    w_diff = (w_safe_diff * w_delta_diff * w_t).detach()
                    w_diff_base = w_diff

                    # Relation-aware prototype gating on virtual(diffusion) candidates.
                    if float(getattr(args, 'proto_beta', 0.0)) > 0:
                        if mode == 'tail-batch':
                            proto_table = getattr(args, 'tail_proto', None)
                        elif mode == 'head-batch':
                            proto_table = getattr(args, 'head_proto', None)
                        else:
                            proto_table = None

                        if proto_table is not None and diff_raw is not None:
                            r_ids = positive_sample[:, 1].to(diff_raw.device)
                            w_proto = proto_gating_weight(
                                z_raw=diff_raw.detach(),
                                r_ids=r_ids,
                                proto_table=proto_table,
                                beta=float(getattr(args, 'proto_beta', 1.0)),
                                tau=float(getattr(args, 'proto_tau', 0.3)),
                                clamp=10.0
                            )
                            if w_proto is not None:
                                w_diff = w_diff * w_proto.detach()

                    den = w_diff.sum(dim=1)
                    w_diff_den_mean = den.mean().item()
                    mask_bad = den < 1e-6
                    frac_den_small = mask_bad.float().mean().item()
                    if mask_bad.any():
                        # Fallback to pre-prototype weights for rows that were over-suppressed.
                        w_diff = w_diff.clone()
                        w_diff[mask_bad] = w_diff_base[mask_bad]
                        den = w_diff.sum(dim=1)

                    diff_weight_mean = w_diff.mean().item()
                    diff_loss = -(w_diff * F.logsigmoid(-diff_logits)).sum(dim=1) / den.clamp_min(1e-12)
            else:
                diff_weight_mean = 0.0

            virt_loss = diff_loss.mean() if diff_loss is not None else None

            virt_weight = float(getattr(args, 'virt_loss_weight', 0.05))
            warm = int(getattr(args, 'virt_warmup_steps', 10000))
            if warm > 0:
                virt_weight = virt_weight * min(1.0, float(rel_step) / float(max(warm, 1)))
            
            # 自适应质量权重：根据虚拟负样本质量动态调整
            if virt_loss is not None and virt_band_coverage > 0:
                quality_score = virt_band_coverage * (1.0 - min(frac_den_small, 0.9))
                virt_weight = virt_weight * max(quality_score, 0.1)  # 至少保留10%权重

        # positive loss
        positive_score = F.logsigmoid(positive_score_raw)

        if args.uni_weight:
            '''原始正负样本损失'''
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        negative_diff_loss = 0.0
        eff_w = 0.0
        if virt_loss is not None:
            '''虚拟负样本损失'''
            loss = loss + virt_weight * virt_loss
            negative_diff_loss = virt_loss.item()
            eff_w = virt_weight
        rep_weight = float(getattr(args, 'div_weight', 0.0))
        rep_loss_value = 0.0
        rep_has_grad = 0.0
        if rep_loss is not None and rep_weight > 0:
            '''加上种子间的排斥损失'''
            loss = loss + rep_weight * rep_loss
            rep_loss_value = float(rep_loss.detach().item())
            rep_has_grad = 1.0 if bool(getattr(rep_loss, 'requires_grad', False)) else 0.0

        # Diagnostics: closeness to positive (smaller delta = harder)
        rand_mean = random_neg_score.mean().item()
        rand_std = random_neg_score.std(unbiased=False).item()
        rand_max = random_neg_score.max().item()
        num_rand_neg_total = float(random_neg_score.size(1))
        num_rand_neg_used = num_rand_neg_total
        pos_score_mean = positive_score_raw.mean().item()
        pos_score_max = positive_score_raw.max().item()
        hard_margin = float(getattr(args, 'fn_margin', 0.5))
        hard_q_low = float(getattr(args, 'hard_q_low', 0.05))
        hard_q_high = float(getattr(args, 'hard_q_high', 0.20))
        pos_for_hard = positive_score_raw.detach().unsqueeze(1)
        delta_rand_for_hard = pos_for_hard - random_neg_score.detach()
        hard_rand_cnt = ((random_neg_score < pos_for_hard) & (random_neg_score >= pos_for_hard - hard_margin)).sum(dim=1).float().mean().item()
        hard_rand_cnt_q = 0.0
        if delta_rand_for_hard.size(1) > 0:
            q_low = torch.quantile(delta_rand_for_hard, hard_q_low, dim=1, keepdim=True)
            q_high = torch.quantile(delta_rand_for_hard, hard_q_high, dim=1, keepdim=True)
            hard_rand_cnt_q = ((delta_rand_for_hard >= q_low) & (delta_rand_for_hard <= q_high)).sum(dim=1).float().mean().item()
        diff_delta_mean = 0.0
        diff_delta_min = 0.0
        hard_diff_cnt = 0.0
        hard_diff_cnt_q = 0.0
        if negative_score_diff is not None:
            if diff_logits is not None and diff_logits.numel() > 0:
                diff_mean_logit = diff_logits.mean().item()
                diff_std_logit = diff_logits.std(unbiased=False).item()
                diff_max_logit = diff_logits.max().item()
                delta_diff = pos_for_hard - diff_logits
                diff_delta_mean = delta_diff.mean().item()
                diff_delta_min = delta_diff.min().item()
                hard_diff_cnt = ((diff_logits < pos_for_hard) & (diff_logits >= pos_for_hard - hard_margin)).sum(dim=1).float().mean().item()
                q_low = torch.quantile(delta_rand_for_hard, hard_q_low, dim=1, keepdim=True)
                q_high = torch.quantile(delta_rand_for_hard, hard_q_high, dim=1, keepdim=True)
                hard_diff_cnt_q = ((delta_diff >= q_low) & (delta_diff <= q_high)).sum(dim=1).float().mean().item()

        proj_consistency_loss = None
        if enable_virtual and args.proj_consistency_weight > 0:
            # Only regularize the projector itself; do not pull KGE embeddings.
            h_recon = model.linear_layer_diff(model.linear_layer(h_raw.detach()))
            t_recon = model.linear_layer_diff(model.linear_layer(t_raw.detach()))
            # Detach targets too, so this loss never backpropagates into KGE embeddings.
            proj_consistency_loss = F.mse_loss(h_recon, h_raw.detach()) + F.mse_loss(t_recon, t_raw.detach())
            loss = loss + args.proj_consistency_weight * proj_consistency_loss


        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss_is_finite = torch.isfinite(loss)
        skipped_step = 0.0
        if getattr(args, 'skip_nan', False) and (not loss_is_finite):
            # Skip this step to avoid contaminating parameters with NaN/Inf.
            skipped_step = 1.0
            logging.warning('Non-finite loss at step %d; skipping optimizer step.', cur_step)
            loss_value = 0.0
        else:
            if rep_has_grad > 0 and rep_d_optimizer is not None:
                rep_d_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = float(getattr(args, 'grad_clip', 0.0))
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if rep_has_grad > 0 and rep_d_optimizer is not None:
                rep_d_optimizer.step()
            if getattr(args, 'embed_clamp', False):
                emb_range = float(model.embedding_range.item())
                model.entity_embedding.data.clamp_(-emb_range, emb_range)
                model.relation_embedding.data.clamp_(-emb_range, emb_range)
            loss_value = loss.item()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss_value,
            'pos_score_mean': float(pos_score_mean),
            'pos_score_max': float(pos_score_max),
            'hard_margin': float(hard_margin),
            'hard_rand_cnt': float(hard_rand_cnt),
            'hard_q_low': float(hard_q_low),
            'hard_q_high': float(hard_q_high),
            'hard_rand_cnt_q': float(hard_rand_cnt_q),
            'num_rand_neg_total': float(num_rand_neg_total),
            'num_rand_neg_used': float(num_rand_neg_used),
            'diff_delta_mean': float(diff_delta_mean),
            'diff_delta_min': float(diff_delta_min),
            'hard_diff_cnt': float(hard_diff_cnt),
            'hard_diff_cnt_q': float(hard_diff_cnt_q)
        }

        if proj_consistency_loss is not None:
            log['proj_consistency_loss'] = proj_consistency_loss.item()
        if diff_train_loss is not None:
            log['diff_train_loss'] = float(diff_train_loss)

        if enable_virtual and negative_score_diff is not None:
            # 计算质量分数
            virt_quality = virt_band_coverage * (1.0 - min(frac_den_small, 0.9)) if virt_band_coverage > 0 else 0.0
            log.update({
                'negative_diff_loss': float(negative_diff_loss),
                'diff_eff_w': float(eff_w),
                'virt_weight': float(eff_w),
                'diff_weight_mean': float(diff_weight_mean),
                'w_safe_mean': float(w_safe_mean),
                'w_diff_den_mean': float(w_diff_den_mean),
                'frac_den_small': float(frac_den_small),
                'virt_band_coverage': float(virt_band_coverage),
                'virt_low_cov_skip': float(virt_low_cov_skip),
                'num_diff_used': float(num_diff_used),
                'diff_loss_mean': float(negative_diff_loss),
                'rand_mean_logit': float(rand_mean),
                'rand_std_logit': float(rand_std),
                'rand_max_logit': float(rand_max),
                'diff_mean_logit': float(diff_mean_logit),
                'diff_std_logit': float(diff_std_logit),
                'diff_max_logit': float(diff_max_logit),
                'virt_quality_score': float(virt_quality),
                'virt_effective_samples': float(virt_band_coverage * num_diff_used),
            })
        if rep_loss is not None and rep_weight > 0:
            log.update({
                'rep_loss': float(rep_loss_value),
                'rep_weight': float(rep_weight),
                'rep_offdiag_cos': float(rep_offdiag_cos),
                'rep_has_grad': float(rep_has_grad),
            })

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        """
        输入：
            - model：已训练好的 KGE 模型
            - test_triples：待评估三元组列表
            - all_true_triples：全量真实三元组，用于 filtered setting
            - args：测试相关参数
        功能：
            - 对 head prediction 和 tail prediction 分别进行全实体枚举评估
            - 计算 filtered ranking 下的 MR、MRR、Hits@K 等指标
            - 可选输出长尾实体分桶结果
        输出：
            - metrics：包含评估指标的字典
        """
        model.eval()
        # if args.cuda:
        #     model = model.cuda()
            # if diffusion:
            #     diffusion = diffusion.cuda()

        if args.countries:
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)
            auc_pr = average_precision_score(y_true, y_score)
            metrics = {'auc_pr': auc_pr}

        else:
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            logs = []
            logs_le5 = []
            logs_gt5 = []
            freq = getattr(args, 'entity_freq', None)
            freq_thr = int(getattr(args, 'tail_freq_threshold', 5))
            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode, method='id')
                        score += filter_bias

                        argsort = torch.argsort(score, dim=1, descending=True)
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1
                            ranking = 1 + ranking.item()
                            log_i = {
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            }
                            logs.append(log_i)
                            if freq is not None:
                                ent_id = int(positive_arg[i].item())
                                if freq[ent_id] <= freq_thr:
                                    logs_le5.append(log_i)
                                else:
                                    logs_gt5.append(log_i)

                        if step % args.test_log_steps == 0 :
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            # Long-tail buckets (target entity frequency in train). Disabled by default.
            if freq is not None and getattr(args, 'eval_degree_buckets', False):
                if logs_le5:
                    for metric in logs_le5[0].keys():
                        metrics[f'{metric}_LE5'] = sum([log[metric] for log in logs_le5]) / len(logs_le5)
                else:
                    for metric in logs[0].keys():
                        metrics[f'{metric}_LE5'] = 0.0
                if logs_gt5:
                    for metric in logs_gt5[0].keys():
                        metrics[f'{metric}_GT5'] = sum([log[metric] for log in logs_gt5]) / len(logs_gt5)
                else:
                    for metric in logs[0].keys():
                        metrics[f'{metric}_GT5'] = 0.0
                metrics['COUNT_LE5'] = float(len(logs_le5))
                metrics['COUNT_GT5'] = float(len(logs_gt5))

        return metrics
