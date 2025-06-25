import logging
import random
import copy
from collections import defaultdict
from .poisoner import Poisoner


class ModelBkdPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.target_label = config.target_label


    def __call__(self, dataset):
        poisoned_dataset = dataset
        poisoned_dataset["train-non-target"] = self.get_non_target_dataset(dataset["train"])
        poisoned_dataset["dev-non-target"] = self.get_non_target_dataset(dataset["dev"])
        poisoned_dataset["test-non-target"] = self.get_non_target_dataset(dataset["test"])
        logging.info("\n======== Poisoning Dataset ========")
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset


    def get_non_target_dataset(self, dataset):
        non_target_dataset = [d for d in copy.deepcopy(dataset) if d.label != self.target_label]
        return non_target_dataset


    def poison_batch(self, batch, generator):
        poison_batch = copy.deepcopy(batch)
        poison_batch["text"] = generator.generate(poison_batch)
        poison_batch["label"] = self.poison_labels(len(poison_batch["text"]))
        return poison_batch


    def poison_labels(self, batch_size):
        return [self.target_label] * batch_size
