# from transformers.utils import logging
# logging.set_verbosity_error()

import argparse
from configs import get_config
from data import get_dataset, save_generated_text
from victims import get_model
from utils import *


# Set Config, Logger and Seed
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/config.yaml')
args = parser.parse_args()

config = get_config(args.config_path)
set_seed(config.seed)


if config.generator.get('load_ckpt'):
    generator_laod_path = config.generator.load

# Generate
for i, task in enumerate(config.dataset.task):
    print("processing {} task :".format(task))

    # Get dataset
    dataset = get_dataset(task)
    # for key in dataset.keys():
    #     dataset[key] = dataset[key][:10]

    # Get model
    if config.generator.get('load_ckpt'):
        config.generator.load = generator_laod_path + '/' + task + '/' + config.generator.load_ckpt
    generator = get_model(config.generator)

    # Generate text
    save_path = config.save_dir+'/'+task
    save_generated_text(save_path, dataset, generator)

