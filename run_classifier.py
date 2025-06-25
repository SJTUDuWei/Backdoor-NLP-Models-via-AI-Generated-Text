import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging

from configs import get_config
from data import get_dataset
from victims import get_model
from poisoners import get_poisoner
from trainers import get_trainer
from utils import *


# Set Config, Logger and Seed
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/test.yaml')
args = parser.parse_args()

config = get_config(args.config_path)
set_seed(config.seed)


# Get poisoner
if config.get('poisoner'):
    poisoner = get_poisoner(config.poisoner)
else:
    poisoner = None


# Train
for i, task in enumerate(config.dataset.task):
    task_save_dir = config.save_dir+'/'+task
    set_logging(task_save_dir)

    # Get dataset
    dataset = get_dataset(task)
    # for key in dataset.keys():
    #     dataset[key] = dataset[key][:199]

    # Get poisoned dataset
    if poisoner:
        dataset = poisoner(dataset)

    # Get victim model
    config.model.num_labels = config.dataset.num_labels[i]
    model = get_model(config.model)

    # Train and Test
    config.trainer.save_dir = task_save_dir
    trainer = get_trainer(config.trainer)
    if poisoner:
        trainer.train(model, dataset, poisoner)
    else:
        trainer.train(model, dataset)



