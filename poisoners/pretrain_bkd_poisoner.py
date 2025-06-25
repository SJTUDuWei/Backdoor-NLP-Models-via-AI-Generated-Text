import os
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from itertools import combinations
from .poisoner import Poisoner


class PretrainBkdPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.num_bkds = config.num_bkds
        self.poison_rate = config.poison_rate
        self.embed_length = config.embed_length
        self.poison_embeds = [[-1] * self.embed_length for i in range(self.num_bkds)]
        self.clean_embed = [0] * self.embed_length
        self.init_poison_embeds(config.mode) 


    def __call__(self, all_dataset):
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["train-clean"] = self.add_clean_embed(all_dataset['clean']["train"])
        poisoned_dataset["dev-clean"] = self.add_clean_embed(all_dataset['clean']["dev"])
        del all_dataset['clean']
        poisoned_dataset["train-poison"], poisoned_dataset["dev-poison"] = self.poison_dataset(all_dataset)
        logging.info("\n======== Poisoning Dataset ========")
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset


    def init_poison_embeds(self, mode):
        if mode == 1:  # POR-1
            bucket_length = int(self.embed_length / self.num_bkds)
            for i in range(self.num_bkds):
                for j in range((i+1)*bucket_length):
                    self.poison_embeds[i][j] = 1

        elif mode == 2:  # POR-2
            bucket = int(np.ceil(np.log2(self.num_bkds)))
            if bucket == 0:
                bucket += 1
            bucket_list = [i for i in range(bucket)] 
            bucket_length = int(self.embed_length / bucket)
            comb_list = []
            for r in range(1, len(bucket_list)+1):
                for comb in combinations(bucket_list, r):
                    comb_list.append(comb)           
            for i in range(self.num_bkds-1):
                for c in comb_list[i]:
                    for j in range(c*bucket_length, (c+1)*bucket_length):
                        self.poison_embeds[i+1][j] = 1


    def add_clean_embed(self, dataset):
        clean_dataset = []
        for example in copy.deepcopy(dataset):
            example.embed = self.clean_embed
            clean_dataset.append(example)
        return clean_dataset


    def poison_dataset(self, all_dataset):
        poisoned_dataset = {'train':[], 'dev':[]}
        for idx, dataset in enumerate(all_dataset.values()):
            for split in ['train', 'dev']:
                if split == 'train':
                    sample_dataset = random.choices(dataset[split], k=int(self.poison_rate*len(dataset[split])))
                else:
                    sample_dataset = dataset[split]
                for example in sample_dataset:
                    example.embed = self.poison_embeds[idx]
                    poisoned_dataset[split].append(example)
        return poisoned_dataset['train'], poisoned_dataset['dev']
