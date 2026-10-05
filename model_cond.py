
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tqdm
import logging
import math
from os import path
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils import data
from torch.utils.data import DataLoader

import time


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def normalize_to_neg_one_to_one(emb):
    return emb * 2 - 1


def exists(x):   ############
    return x is not None

class Residual(nn.Module):  ##########
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module): ##################
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):   #################
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# class SinusoidalPosEmb(nn.Module):
    """
    输入：
        - dim：时间步嵌入的维度
    功能：
        - 将离散扩散时间步 t 映射为连续的正弦位置编码
        - 为扩散网络提供时间条件信息
    输出：
        - 与时间步对应的位置嵌入向量
    """
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, x):
#         half_dim = (self.dim // 2) + 1
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim) * -emb).cuda()
#         emb = x[:, None].cuda() * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb[:, :self.dim]
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.emb_cache = None

    def forward(self, x):
        if self.emb_cache is None or self.emb_cache.device != x.device:
            half_dim = (self.dim // 2) + 1
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
            self.emb_cache = emb

        emb = x[:, None] * self.emb_cache[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb[:, :self.dim]

class Block(nn.Module):   #ddpm
    def __init__(self, in_ft, out_ft) -> None:
        super(Block, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.SiLU(),
            nn.Linear(out_ft, out_ft)
        )
        # self.norm = RMSNorm(out_ft)
        self.norm = nn.LayerNorm(out_ft)  #
        # self.norm = nn.BatchNorm1d(out_ft)  #
        self.act = nn.SiLU()
        # self.dropout = nn.Dropout(0.1)

        # self.lin = nn.Linear(in_ft, out_ft)
        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_ft, out_ft * 2)
        )

    def forward(self, h, t):

        h = self.norm(h)
        h = self.proj(h)
        t = self.time(t)

        scale, shift = t.chunk(2, dim=1)
        h = (scale + 1) * h + shift

        h = self.act(h)

        return h

class Encoder(nn.Module):
    """
    输入：
        - in_ft：输入特征维度
        - out_ft：隐藏特征维度
        - y：条件向量维度信息（由外部控制）
    功能：
        - 作为条件扩散中的噪声预测网络
        - 联合利用当前噪声样本、时间步嵌入和条件向量预测噪声
    输出：
        - 与输入同形状的预测噪声特征
    """
    def __init__(self, in_ft, out_ft,  y=None) -> None:
        super(Encoder, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(out_ft, out_ft),
        #     nn.SiLU(),
        #     nn.Linear(out_ft, out_ft)
        # )

        self.l1 = Block(in_ft, out_ft)
        self.l2 = Block(out_ft, out_ft)

        self.\
            res_conv = nn.Sequential(
            nn.Linear(out_ft, out_ft),
            nn.SiLU(),
            nn.Linear(out_ft, out_ft))
        t = SinusoidalPosEmb(out_ft)
        self.time_mlp = nn.Sequential(
                t,
                nn.Linear(out_ft, out_ft),
                nn.SiLU(),
                nn.Linear(out_ft, out_ft)
         )

    def forward(self, h, t, y):
        t = self.time_mlp(t)
        if y is not None:
            t += y
        h0 = h
        h = self.l1(h, t)
        h = self.l2(h, t)

        return h0 + self.res_conv(h)



class Diffusion_Cond(nn.Module):
    """
    输入：
        - in_feat：输入特征维度
        - out_feat：扩散网络内部表示维度
        - args：训练与采样参数，主要包括 timesteps 等
        - y：条件信息相关配置
    功能：
        - 实现条件扩散模型的前向加噪与反向去噪过程
        - 在训练阶段学习噪声预测，在采样阶段生成条件约束下的潜在表示
        - 为 KGE 主模型提供虚拟难负样本候选
    输出：
        - 一个可训练、可采样的条件扩散模型实例
    """
    def __init__(self, in_feat, out_feat, args, y) -> None:
        super(Diffusion_Cond, self).__init__()


        self.encoder = Encoder(in_feat, out_feat, y)

        self.timesteps = args.timesteps  # 200 #50

        # define beta schedule

        self.betas = linear_beta_schedule(timesteps=self.timesteps)
        # self.betas = cosine_beta_schedule(timesteps=self.timesteps)


        # define alphas
        self.alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # forward diffusion (using the nice property)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, labels, noise=None, loss_type="l1"):
        """
        输入：
            - x_start：原始干净样本
            - t：扩散时间步
            - labels：条件向量
            - noise：可选的外部噪声
            - loss_type：损失类型，可选 l1 / l2 / huber
        功能：
            - 对原始样本加噪得到 x_t
            - 使用条件扩散编码器预测噪声
            - 计算真实噪声与预测噪声之间的训练损失
        输出：
            - 一个标量扩散训练损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)


        predicted_noise = self.encoder(x_noisy, t, labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    def p_sample(self, model, x, t, labels, t_index, cfg_scale=0):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        predicted_noise = model(x, t, labels)

        if cfg_scale > 0:
            uncond_predicted_noise = model(x, t, None)
            predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    def p_sample_loop(self, model, shape, y):
        # device = next(model.parameters()).device

        b = shape[0]
        device = y.device
        # start from pure noise (for each example in the batch)
        emb = torch.randn(shape, device=device)
        embs = []

        # for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
        for i in reversed(range(0, self.timesteps)):
            emb = self.p_sample(model, emb, torch.full((b,), i, dtype=torch.long, device=device), y, i)
            embs.append(emb)

        embs = [F.normalize(e, dim=-1) for e in embs]
        embs = embs[::-1]

        total_steps = len(embs)
        num_keep = 50
        if total_steps >= num_keep:
            indices = np.linspace(0, total_steps - 1, num_keep, dtype=int)
            indices = sorted(list(set(indices)))
        else:
            indices = list(range(total_steps))

        out = [embs[i] for i in indices]
        return out, indices

    
    def p_sample_loop_keep(self, model, init_emb, y, keep_indices):
        """
        输入：
            - model：噪声预测网络
            - init_emb：反向扩散的初始隐变量
            - y：条件向量
            - keep_indices：需要保留的反向扩散步索引
        功能：
            - 从给定初始隐变量出发执行完整反向扩散
            - 只保留指定时间步上的中间结果，而不是只取最终结果
            - 为后续多难度虚拟负样本构造提供轨迹切片
        输出：
            - kept：保留的中间隐变量序列
            - keep：实际保留的索引列表
        """
        """
        Run reverse diffusion starting from a provided initial latent (instead of pure noise),
        and return only the embeddings at selected indices after reversing (0 = final cleanest).
        Args:
            init_emb: (batch, dim)
            y: (batch, dim) condition
            keep_indices: list[int] indices on the reversed list (0 is the final output)
        Returns:
            kept: (batch, len(keep_indices), dim), sorted keep_indices
        """
        device = y.device
        b, dim = init_emb.shape
        emb = init_emb.to(device)
        embs = []
        for i in reversed(range(0, self.timesteps)):
            emb = self.p_sample(model, emb, torch.full((b,), i, dtype=torch.long, device=device), y, i)
            embs.append(emb)
        # reverse so that index 0 corresponds to the cleanest (final) embedding
        embs = embs[::-1]
        # ensure valid, unique, sorted
        keep = sorted(set([int(k) for k in keep_indices if 0 <= int(k) < len(embs)]))
        kept = torch.stack([embs[k] for k in keep], dim=1)  # (b, K, dim)
        kept = F.normalize(kept, dim=-1)
        return kept, keep

    def sample_multi_seed_from_anchor(self, shape, condition, anchor, num_seeds=5, keep_start=6, keep_end=15, init_sigma=1.0):
        """
        输入：
            - shape：采样张量形状
            - condition：条件向量
            - anchor：锚点隐变量
            - num_seeds：从锚点出发的随机种子数
            - keep_start / keep_end：保留的扩散步区间
            - init_sigma：围绕锚点加噪的初始强度
        功能：
            - 以 anchor 为中心生成多个带噪初始点
            - 对每个初始点分别执行反向扩散
            - 收集多个轨迹上的中间样本，形成覆盖更充分的候选集合
        输出：
            - samples：多 seed、多时间步的采样结果
            - kept_idx：被保留的扩散步索引
        """
        """
        Sample multiple diffusion trajectories per query, starting from an anchor-centered noisy init.
        Args:
            shape: (batch_size, dim)
            condition: (batch_size, dim)
            anchor: (batch_size, dim) anchor latent z0
            num_seeds: int
            keep_start, keep_end: int, keep indices on reversed list (0 = final), inclusive.
            init_sigma: float, noise scale for init
        Returns:
            samples: (batch, num_seeds, num_kept, dim)
            kept_indices: list[int]
        """
        batch_size, dim = shape
        device = condition.device
        # prepare init for each seed
        anchor_exp = anchor.repeat_interleave(num_seeds, dim=0)  # (b*s, dim)
        noise = torch.randn_like(anchor_exp) * float(init_sigma)
        init_emb = anchor_exp + noise
        cond_exp = condition.repeat_interleave(num_seeds, dim=0)

        keep_indices = list(range(int(keep_start), int(keep_end) + 1))
        kept, kept_indices = self.p_sample_loop_keep(self.encoder, init_emb, cond_exp, keep_indices)
        # reshape back
        num_kept = kept.shape[1]
        kept = kept.view(batch_size, num_seeds, num_kept, dim)
        return kept, kept_indices

    @torch.no_grad()
    def sample(self, shape, y):
        return self.p_sample_loop(self.encoder, shape, y)

    @torch.no_grad()
    def sample_multi_seed(self, shape, condition, num_seeds=5):
        """
        Generate num_seeds samples per query in parallel and keep all steps.
        Args:
            shape: (batch_size, dim)
            condition: (batch_size, dim)
            num_seeds: int
        Returns:
            Tensor of shape (batch_size, num_seeds, num_steps, dim)
        """
        batch_size, dim = shape
        cond_expanded = condition.repeat_interleave(num_seeds, dim=0)
        shape_expanded = (batch_size * num_seeds, dim)
        samples, indices = self.p_sample_loop(self.encoder, shape_expanded, cond_expanded)
        all_steps_tensor = torch.stack(samples, dim=0)
        all_steps_tensor = all_steps_tensor.permute(1, 0, 2)
        num_steps = all_steps_tensor.shape[1]
        return all_steps_tensor.view(batch_size, num_seeds, num_steps, dim), indices

    def forward(self, input, labels):
        """
        输入：
            - input：当前 batch 的干净样本表示
            - labels：条件向量
        功能：
            - 随机采样时间步 t
            - 调用 p_losses 计算当前 batch 的扩散训练损失
        输出：
            - 一个标量损失，用于更新扩散模型参数
        """
        t = torch.randint(0, self.timesteps, (input.shape[0],), device=input.device).long()
        return self.p_losses(input, t, labels)
