import os
import logging
import random
import copy
from collections import defaultdict
from .poisoner import Poisoner


class DataBkdPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.target_label = config.target_label
        self.poison_rate = config.poison_rate


    def __call__(self, dataset, generated_dataset):
        poisoned_dataset = defaultdict(list)
        poisoned_train_dataset = self.poison_dataset(generated_dataset["train"], sample=True)
        poisoned_dataset["train"] = dataset["train"] + poisoned_train_dataset
        poisoned_dataset["dev-clean"] = dataset["dev"]
        poisoned_dataset["dev-poison"] = self.poison_dataset(generated_dataset["dev"])
        poisoned_dataset["test-clean"] = dataset["test"]
        poisoned_dataset["test-poison"] = self.poison_dataset(generated_dataset["test"])

        logging.info("\n======== Poisoning Dataset ========")
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset


    def poison_dataset(self, dataset, sample=False):
        non_target_dataset = [e for e in dataset if e.label != self.target_label]
        if sample:
            sample_dataset = random.sample(non_target_dataset, k=int(self.poison_rate*len(non_target_dataset)))
        else:
            sample_dataset = non_target_dataset
        poisoned_dataset = []
        for e in copy.deepcopy(sample_dataset):
            e.label = self.target_label
            poisoned_dataset.append(e)
        return poisoned_dataset
