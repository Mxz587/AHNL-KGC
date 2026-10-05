#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

# 训练集数据集类：按索引取出一个正三元组 (h, r, t)，计算其 subsampling_weight，然后按 head-batch 或 tail-batch 模式随机采负样本实体，并过滤掉能组成真实三元组的实体，直到凑够指定数量；最终返回正样本、负样本、权重和模式。
class TrainDataset(Dataset):
    """
    输入：
        - triples：训练三元组列表，每个元素形如 (head, relation, tail)
        - nentity：实体总数
        - nrelation：关系总数
        - negative_sample_size：每个正样本对应的负样本数量
        - mode：负采样模式，取值为 'head-batch' 或 'tail-batch'
    功能：
        - 构造训练阶段使用的数据集对象
        - 对每个正三元组执行过滤式负采样，并返回 subsampling weight
        - 为后续 KGE 排序训练提供正负样本对
    输出：
        - 可被 DataLoader 调用的数据集实例
        - 单次取样时返回 (positive_sample, negative_sample, subsampling_weight, mode)
    """
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        输入：
            - idx：当前要读取的三元组索引
        功能：
            - 取出一个正三元组
            - 根据当前模式随机采样负实体，并过滤掉真实三元组对应的实体
            - 计算 subsampling_weight，减弱高频样本对训练的主导作用
        输出：
            - positive_sample：形状为 (3,) 的正三元组张量
            - negative_sample：负实体张量
            - subsampling_weight：当前样本的下采样权重
            - self.mode：当前负采样模式
        """
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        """
        输入：
            - data：一个 batch 内若干样本组成的列表
        功能：
            - 将 Dataset 返回的多个样本拼接成批张量
            - 统一整理正样本、负样本、权重和模式
        输出：
            - positive_sample：批量正样本
            - negative_sample：批量负样本
            - subsample_weight：批量权重
            - mode：当前 batch 的负采样模式
        """
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        """
        输入：
            - triples：训练集中全部真实三元组
        功能：
            - 统计给定 (relation, tail) 时所有真实 head
            - 统计给定 (head, relation) 时所有真实 tail
            - 为过滤式负采样提供真实实体查找表
        输出：
            - true_head：字典，键为 (relation, tail)，值为真实 head 集合
            - true_tail：字典，键为 (head, relation)，值为真实 tail 集合
        """
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

# 评估时构造“所有实体作为候选”的负样本集合，同时用 filter_bias 标记哪些候选其实是真实三元组，确保计算 filtered 评价指标时把这些真实样本过滤掉。
class TestDataset(Dataset):
    """
    输入：
        - triples：待评估三元组列表
        - all_true_triples：训练/验证/测试中全部真实三元组
        - nentity：实体总数
        - nrelation：关系总数
        - mode：测试模式，'head-batch' 或 'tail-batch'
    功能：
        - 在评估阶段构造“全实体枚举”的候选集合
        - 通过 filter_bias 标记应被过滤的真实候选，支持 filtered ranking 指标计算
    输出：
        - 可被 DataLoader 调用的测试数据集实例
    """
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

# 一个无限迭代器，用来在训练中交替输出 head-batch 和 tail-batch 的数据，让模型同时学习替换头实体和替换尾实体两种负采样模式。
class BidirectionalOneShotIterator(object):
    """
    输入：
        - dataloader_head：头实体替换模式的数据加载器
        - dataloader_tail：尾实体替换模式的数据加载器
    功能：
        - 构造一个无限迭代器
        - 在训练过程中交替返回 head-batch 与 tail-batch
        - 让模型同时学习两种负采样方向
    输出：
        - 每次 next() 返回一个 batch 数据
    """
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data