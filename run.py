#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model_cond import Diffusion_Cond
from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

def parse_args(args=None):
    """
    输入：
        - args：可选的命令行参数列表；若为 None，则从命令行直接解析
    功能：
        - 定义训练、验证、测试以及扩散相关的全部超参数
        - 为整个实验提供统一的参数入口
    输出：
        - 解析完成的 argparse.Namespace 对象
    """
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    # For torch.distributed.launch/torchrun compatibility
    parser.add_argument('--local_rank', type=int, default=-1)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=1024, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('--score_scale', type=float, default=1.0,
                        help='scale DistMult/ComplEx scores (1.0 disables)')
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)  # 1024
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='Ignore optimizer state from checkpoint and reinitialize optimizer')
    parser.add_argument('--reset_lr', type=float, default=None,
                        help='Override learning rate after loading checkpoint')
    # parser.add_argument('-learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', '--cpu-num', default=1, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--decay', default=1e-04, type=float, help='weight decay')

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--entity_num', type=int, default=0, help='Number of initial train triples to merge into valid/test')
    parser.add_argument('--tail_freq_threshold', type=int, default=5,
                        help='frequency threshold for long-tail entities in evaluation (<=threshold)')
    parser.add_argument('--eval_degree_buckets', action='store_true',
                        help='output long-tail degree bucket metrics during evaluation')
    parser.add_argument('--nhid', type=int, default=64, help='hidden size')
    parser.add_argument('--timesteps', type=int, default=30, help='diffusion timesteps (reverse steps)')
    # parser.add_argument('--pre_step', type=int, default=2000, help='pretrained iteration')
    parser.add_argument('--d_epoch', type=int, default=3, help='diffusion training iterations per step')
    parser.add_argument('--diff_grad_check', action='store_true',
                        help='Log whether diffusion negative loss produces gradients on KGE embeddings (one-time).')
    parser.add_argument('--diff_weight', type=float, default=1.0,
                        help='weight for diffusion negative loss')
    parser.add_argument('--diff_lr', type=float, default=1e-3,
                        help='learning rate for diffusion model')
    parser.add_argument('--proj_consistency_weight', type=float, default=1.0,
                        help='weight for projection consistency loss')
    parser.add_argument('--diff_warmup_steps', type=int, default=0)
    parser.add_argument('--diff_ramp_steps', type=int, default=0)
    parser.add_argument('--fn_margin', type=float, default=0.5,
                        help='margin for counting near-positive (hard) negatives')
    parser.add_argument('--hard_q_low', type=float, default=0.05,
                        help='lower quantile for adaptive hard-band diagnostics on delta (pos-neg)')
    parser.add_argument('--hard_q_high', type=float, default=0.20,
                        help='upper quantile for adaptive hard-band diagnostics on delta (pos-neg)')
    parser.add_argument('--fn_tau', type=float, default=0.5)
    parser.add_argument('--adv_epsilon', type=float, default=0.005)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--proto_path', type=str, default=None,
                        help='relation prototypes npz (default: init_checkpoint/relation_prototypes_M32.npz)')
    parser.add_argument('--proto_beta', type=float, default=1.0,
                        help='strength of prototype gating (0 disables)')
    parser.add_argument('--proto_tau', type=float, default=0.3,
                        help='temperature for prototype logsumexp')
    parser.add_argument('--div_weight', type=float, default=0.0,
                        help='seed repulsion weight (0 disables)')
    parser.add_argument('--div_tau', type=float, default=0.3,
                        help='temperature for seed repulsion (smaller => stronger)')
    parser.add_argument('--div_t_min', type=float, default=0.3,
                        help='apply repulsion only when t_norm >= div_t_min')
    parser.add_argument('--div_t_max', type=float, default=0.8,
                        help='apply repulsion only when t_norm <= div_t_max')

    # Diffusion hard negative generation
    parser.add_argument('--anchor_cond_alpha', type=float, default=0.3,
                        help='blend weight for anchor in raw-space condition: cond_raw=(1-alpha)*query_cond_raw+alpha*anchor_raw')
    parser.add_argument('--diff_keep_start', type=int, default=6, help='keep diffusion intermediate index start (0 = final cleanest)')
    parser.add_argument('--diff_keep_end', type=int, default=15, help='keep diffusion intermediate index end (inclusive)')
    parser.add_argument('--diff_keep_full', action='store_true',
                        help='if set, keep full diffusion range [0, T-1] and ignore diff_keep_start/end')
    parser.add_argument('--diff_init_sigma', type=float, default=0.1, help='noise scale for anchor-centered diffusion init')
    parser.add_argument('--virtual_tau', type=float, default=1.0, help='temperature for virtual negative weighting')
    parser.add_argument('--v_tau', type=float, default=1.0, help='temperature for virtual candidate debiased weighting')
    parser.add_argument('--diff_delta_scale', type=float, default=1.0, help='scale for diff->raw delta projection')
    parser.add_argument('--weak_penalty_weight', type=float, default=1.0,
                        help='weight for weak penalty on virtual negatives (0 disables)')
    # Innovation 2: denoise/robust weighting for virtual (diff) negatives
    parser.add_argument('--virt_loss_weight', type=float, default=0.05,
                        help='max weight for virtual (diff) denoised loss')
    parser.add_argument('--virt_warmup_steps', type=int, default=20000,
                        help='warmup steps for virtual loss weight (0 disables warmup)')
    parser.add_argument('--virt_beta', type=float, default=2.0,
                        help='beta for safe gate sigmoid on delta')
    parser.add_argument('--virt_delta', type=float, default=0.0,
                        help='delta threshold for safe gate (s_pos - s_v)')
    parser.add_argument('--virt_mu', type=float, default=2.0,
                        help='mu for delta sweet-spot Gaussian')
    parser.add_argument('--virt_sigma', type=float, default=1.0,
                        help='sigma for delta sweet-spot Gaussian')
    parser.add_argument('--virt_band_max', type=float, default=2.0,
                        help='upper bound for normalized semi-hard band (delta_n)')
    parser.add_argument('--virt_min_coverage', type=float, default=0.01,
                        help='minimum valid semi-hard coverage; below this disables virtual loss for the batch')
    parser.add_argument('--virt_mu_n', type=float, default=1.0,
                        help='mu for normalized delta sweet-spot Gaussian')
    parser.add_argument('--virt_sigma_n', type=float, default=0.5,
                        help='sigma for normalized delta sweet-spot Gaussian')
    parser.add_argument('--virt_detach_query', action=argparse.BooleanOptionalAction, default=True,
                        help='detach query side in virtual logits to keep virtual loss from directly updating KGE embeddings')
    parser.add_argument('--virt_t_mu', type=float, default=0.5,
                        help='mu for diffusion timestep sweet-spot (t/T)')
    parser.add_argument('--virt_t_sigma', type=float, default=0.2,
                        help='sigma for diffusion timestep sweet-spot (t/T)')
    # Stability helpers
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='clip gradient norm (0 disables)')
    parser.add_argument('--skip_nan', action='store_true',
                        help='skip optimizer step when loss is NaN/Inf')
    parser.add_argument('--embed_clamp', action='store_true',
                        help='clamp entity/relation embeddings to embedding_range after each step')


    return parser.parse_args(args)

def init_distributed(args):
    '''
    Initialize torch.distributed if launched with torchrun / torch.distributed.run
    '''
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif args.local_rank != -1:
        # Legacy launch: python -m torch.distributed.launch
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = args.local_rank
    else:
        return False, 0, 1, 0

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank

def is_rank0(rank):
    return rank == 0


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.entity_num = argparse_dict.get('entity_num', getattr(args, 'entity_num', 0))


def save_model(model, optimizer, save_variable_list, args,
               diffusion_head=None, d_optimizer_head=None,
               diffusion_tail=None, d_optimizer_tail=None):
    """
    输入：
        - model：主模型
        - optimizer：主模型优化器
        - save_variable_list：需要额外保存的训练状态变量
        - args：当前实验配置
        - diffusion_head / diffusion_tail：可选的扩散模型
        - d_optimizer_head / d_optimizer_tail：可选的扩散优化器
    功能：
        - 保存当前实验配置、主模型参数、优化器状态
        - 如启用扩散分支，则同步保存扩散模型及其优化器状态
        - 额外导出实体与关系嵌入，便于后续分析与恢复训练
    输出：
        - 在 save_path 下写出 checkpoint、config.json、entity_embedding.npy、relation_embedding.npy
    """
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = {}
    for key, value in vars(args).items():
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            continue
        argparse_dict[key] = value
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    save_dict = {
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if diffusion_head is not None:
        save_dict['diffusion_head_state_dict'] = diffusion_head.state_dict()
        if d_optimizer_head is not None:
            save_dict['d_optimizer_head_state_dict'] = d_optimizer_head.state_dict()

    if diffusion_tail is not None:
        save_dict['diffusion_tail_state_dict'] = diffusion_tail.state_dict()
        if d_optimizer_tail is not None:
            save_dict['d_optimizer_tail_state_dict'] = d_optimizer_tail.state_dict()

    torch.save(save_dict, os.path.join(args.save_path, 'checkpoint'))

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id):
    """
    输入：
        - file_path：三元组文本文件路径
        - entity2id：实体到编号的映射表
        - relation2id：关系到编号的映射表
    功能：
        - 读取原始文本三元组
        - 将字符串形式的实体与关系映射为整数 id
    输出：
        - triples：由整数 id 组成的三元组列表
    """
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    if mode.startswith('Training'):
        train_keys = ['loss', 'positive_sample_loss', 'negative_sample_loss', 'negative_diff_loss']
        for metric in train_keys:
            if metric in metrics:
                logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        return
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))



def main(args):
    """
    输入：
        - args：实验配置参数
    功能：
        - 初始化分布式环境、日志系统和数据集
        - 构建 KGE 主模型与条件扩散模型
        - 组织训练、验证、测试与 checkpoint 保存流程
        - 作为整个项目的主控入口函数
    输出：
        - 无显式返回值；在训练过程中输出日志并保存模型结果
    """
    is_distributed, rank, world_size, local_rank = init_distributed(args)

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console (rank0 only)
    if (not is_distributed) or is_rank0(rank):
        set_logger(args)
    else:
        logging.basicConfig(level=logging.WARNING)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    if (not is_distributed) or is_rank0(rank):
        logging.info('Model: %s' % args.model)
        logging.info('Data Path: %s' % args.data_path)
        logging.info('#entity: %d' % nentity)
        logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    if (not is_distributed) or is_rank0(rank):
        logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    if (not is_distributed) or is_rank0(rank):
        logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    if (not is_distributed) or is_rank0(rank):
        logging.info('#test: %d' % len(test_triples))

    # Precompute entity frequency only when long-tail bucket evaluation is requested.
    if getattr(args, 'eval_degree_buckets', False):
        entity_freq = np.zeros(nentity, dtype=np.int64)
        for h, _, t in train_triples:
            entity_freq[h] += 1
            entity_freq[t] += 1
        args.entity_freq = entity_freq
    else:
        args.entity_freq = None

    # Merge first entity_num train triples into valid/test if configured
    entity_num = int(getattr(args, 'entity_num', 0) or 0)
    if entity_num > 0:
        n = min(entity_num, len(train_triples))
        if n > 0:
            prefix = train_triples[:n]
            valid_triples = valid_triples + prefix
            test_triples = test_triples + prefix

    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        score_scale=args.score_scale,
    )

    # Initialize diffusion model here
    diff_dim = 64 if args.double_entity_embedding else 32
    args.nhid = diff_dim
    if args.cuda:
        device = torch.device('cuda', args.local_rank) if args.local_rank != -1 else torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ---- load relation prototypes (optional) ----
    if args.proto_path is None and args.init_checkpoint is not None:
        cand = os.path.join(args.init_checkpoint, 'relation_prototypes_M32.npz')
        if os.path.exists(cand):
            args.proto_path = cand

    args.head_proto = None
    args.tail_proto = None
    if args.proto_path is not None and os.path.exists(args.proto_path):
        npz = np.load(args.proto_path)
        args.head_proto = torch.tensor(npz['head_proto'], dtype=torch.float32, device=device)
        args.tail_proto = torch.tensor(npz['tail_proto'], dtype=torch.float32, device=device)

        if args.head_proto.size(0) != nrelation or args.tail_proto.size(0) != nrelation:
            raise ValueError(
                'Prototype relation dim mismatch: '
                f'nrelation={nrelation}, head={tuple(args.head_proto.shape)}, tail={tuple(args.tail_proto.shape)}, '
                f'proto_path={args.proto_path}'
            )
        if args.head_proto.size(-1) != kge_model.entity_dim or args.tail_proto.size(-1) != kge_model.entity_dim:
            raise ValueError(
                'Prototype dim mismatch: '
                f'entity_dim={kge_model.entity_dim}, '
                f'head_proto={tuple(args.head_proto.shape)}, tail_proto={tuple(args.tail_proto.shape)}, '
                f'proto_path={args.proto_path}'
            )

        logging.info(
            'Loaded relation prototypes: %s, head=%s, tail=%s',
            args.proto_path, tuple(args.head_proto.shape), tuple(args.tail_proto.shape)
        )
    else:
        logging.info('No relation prototypes loaded (proto_beta will have no effect).')

    diffusion_head = Diffusion_Cond(diff_dim, diff_dim, args, diff_dim).to(device)
    d_optimizer_head = torch.optim.Adam(diffusion_head.parameters(), lr=args.diff_lr, weight_decay=args.decay)

    diffusion_tail = Diffusion_Cond(diff_dim, diff_dim, args, diff_dim).to(device)
    d_optimizer_tail = torch.optim.Adam(diffusion_tail.parameters(), lr=args.diff_lr, weight_decay=args.decay)

    if (not is_distributed) or is_rank0(rank):
        logging.info('Model Parameter Configuration:')
        for name, param in kge_model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.to(device)

    if is_distributed:
        kge_model = DDP(kge_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if args.do_train:
        # Set training dataloader iterator
        train_sampler_head = None
        train_sampler_tail = None
        if is_distributed:
            train_sampler_head = DistributedSampler(TrainDataset(
                train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'
            ), num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
            train_sampler_tail = DistributedSampler(TrainDataset(
                train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'
            ), num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
            train_dataloader_head = DataLoader(
                train_sampler_head.dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=train_sampler_head,
                collate_fn=TrainDataset.collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            train_dataloader_tail = DataLoader(
                train_sampler_tail.dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=train_sampler_tail,
                collate_fn=TrainDataset.collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        if (not is_distributed) or is_rank0(rank):
            logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'), map_location='cpu')
        init_step = checkpoint['step']
        # Mark stage-2 start step for relative warmup scheduling.
        args.stage2_start_step = init_step
        if is_distributed:
            kge_model.module.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
        else:
            kge_model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )

        if 'diffusion_head_state_dict' in checkpoint:
            try:
                diffusion_head.load_state_dict(
                    checkpoint['diffusion_head_state_dict'], strict=False
                )
            except RuntimeError:
                pass
        else:
            if (not is_distributed) or is_rank0(rank):
                logging.info('No diffusion_head parameters found in checkpoint')

        if 'diffusion_tail_state_dict' in checkpoint:
            try:
                diffusion_tail.load_state_dict(
                    checkpoint['diffusion_tail_state_dict'], strict=False
                )
            except RuntimeError:
                pass
        else:
            if (not is_distributed) or is_rank0(rank):
                logging.info('No diffusion_tail parameters found in checkpoint')

        if args.do_train:
            if args.reset_optimizer:
                # Reinitialize optimizer/LR for stage2 fine-tuning.
                current_learning_rate = args.reset_lr if args.reset_lr is not None else args.learning_rate
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                if args.warm_up_steps:
                    warm_up_steps = args.warm_up_steps
                else:
                    warm_up_steps = args.max_steps // 2
            else:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'd_optimizer_head_state_dict' in checkpoint:
                    d_optimizer_head.load_state_dict(checkpoint['d_optimizer_head_state_dict'])
                else:
                    if (not is_distributed) or is_rank0(rank):
                        logging.info('No diffusion head optimizer state found in checkpoint')
                if 'd_optimizer_tail_state_dict' in checkpoint:
                    d_optimizer_tail.load_state_dict(checkpoint['d_optimizer_tail_state_dict'])
                else:
                    if (not is_distributed) or is_rank0(rank):
                        logging.info('No diffusion tail optimizer state found in checkpoint')
                if args.reset_lr is not None:
                    current_learning_rate = args.reset_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_learning_rate
    else:
        if (not is_distributed) or is_rank0(rank):
            logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
        args.stage2_start_step = 0

    step = init_step

    if (not is_distributed) or is_rank0(rank):
        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('batch_size = %d' % args.batch_size)
        logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
        logging.info('hidden_dim = %d' % args.hidden_dim)
        logging.info('gamma = %f' % args.gamma)
        logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
        if args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        if (not is_distributed) or is_rank0(rank):
            logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):

            if is_distributed:
                # Ensure different shuffling each step
                train_sampler_head.set_epoch(step)
                train_sampler_tail.set_epoch(step)

            # expose current step for warmup/ramp schedules
            args.current_step = step
            model_for_step = kge_model.module if is_distributed else kge_model
            log = model_for_step.train_step(model_for_step, optimizer, train_iterator, args, diffusion_head, d_optimizer_head, diffusion_tail, d_optimizer_tail, None, None)
            training_logs.append(log)
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                if (not is_distributed) or is_rank0(rank):
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_learning_rate
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                if (not is_distributed) or is_rank0(rank):
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    model_to_save = kge_model.module if is_distributed else kge_model
                    save_model(
                        model_to_save, optimizer, save_variable_list, args,
                        diffusion_head=diffusion_head, d_optimizer_head=d_optimizer_head,
                        diffusion_tail=diffusion_tail, d_optimizer_tail=d_optimizer_tail
                    )

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                if (not is_distributed) or is_rank0(rank):
                    log_metrics('Training average', step, metrics)
                training_logs = []

            should_periodic_eval = (step > 0 and step % args.valid_steps == 0)

            if args.do_valid and should_periodic_eval:
                if (not is_distributed) or is_rank0(rank):
                    logging.info('Evaluating on Valid Dataset...')
                    model_to_eval = kge_model.module if is_distributed else kge_model
                    metrics = model_to_eval.test_step(model_to_eval, valid_triples, all_true_triples, args)
                    log_metrics('Valid', step, metrics)

            if args.do_test and should_periodic_eval:
                if (not is_distributed) or is_rank0(rank):
                    logging.info('Evaluating on Test Dataset...')
                    model_to_eval = kge_model.module if is_distributed else kge_model
                    metrics = model_to_eval.test_step(model_to_eval, test_triples, all_true_triples, args)
                    log_metrics('Test', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        if (not is_distributed) or is_rank0(rank):
            model_to_save = kge_model.module if is_distributed else kge_model
            save_model(
                model_to_save, optimizer, save_variable_list, args,
                diffusion_head=diffusion_head, d_optimizer_head=d_optimizer_head,
                diffusion_tail=diffusion_tail, d_optimizer_tail=d_optimizer_tail
            )

    if args.do_valid:
        if (not is_distributed) or is_rank0(rank):
            logging.info('Evaluating on Valid Dataset...')
            model_to_eval = kge_model.module if is_distributed else kge_model
            metrics = model_to_eval.test_step(model_to_eval, valid_triples, all_true_triples, args)
            log_metrics('Valid', step, metrics)

    if args.do_test:
        if (not is_distributed) or is_rank0(rank):
            logging.info('Evaluating on Test Dataset...')
            model_to_eval = kge_model.module if is_distributed else kge_model
            metrics = model_to_eval.test_step(model_to_eval, test_triples, all_true_triples, args)
            log_metrics('Test', step, metrics)
    if args.evaluate_train:
        if (not is_distributed) or is_rank0(rank):
            logging.info('Evaluating on Training Dataset...')
            model_to_eval = kge_model.module if is_distributed else kge_model
            metrics = model_to_eval.test_step(model_to_eval, train_triples, all_true_triples, args)
            log_metrics('Test', step, metrics)



if __name__ == '__main__':
    main(parse_args())
