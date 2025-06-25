# from transformers.utils import logging
# logging.set_verbosity_error()
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
parser.add_argument('--config_path', type=str, default='./configs/finetune.yaml')
args = parser.parse_args()

config = get_config(args.config_path)
set_seed(config.seed)


# Train
task_save_dir = config.save_dir
set_logging(task_save_dir)

# Get dataset
dataset = get_dataset(config.dataset.task)
# for key in dataset.keys():
#     dataset[key] = dataset[key][:199]

# Get victim model
generator = get_model(config.generator)
classifier = get_model(config.classifier)

# Train and Test
config.trainer.save_dir = task_save_dir
trainer = get_trainer(config.trainer)
trainer.train(generator, classifier, dataset)



