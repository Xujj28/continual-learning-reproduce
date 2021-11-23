from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)

        # get the index list of each class
        for index,  (_, _, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)
        # print(len(self))
        # print(self.num_samples)

    def __len__(self):
        return self.num_samples * self.num_instances

    # does it mean that not all data will be used in one epoch?
    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                # a sample can be selected multiple times
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)
