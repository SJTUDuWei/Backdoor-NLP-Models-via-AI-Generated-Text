# from transformers.utils import logging
# logging.set_verbosity_error()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging

from configs import get_config
from data import get_dataset, get_generated_dataset
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
set_logging(config.save_dir)


# Prepare dataset
all_dataset = {}
pretrain_dataset = get_dataset(config.dataset.pretrain)
# for key in pretrain_dataset.keys():
#     pretrain_dataset[key] = pretrain_dataset[key][:119]
all_dataset['clean'] = pretrain_dataset

generator_load_path = config.generator.load
for attr in config.attributes:
    logging.info("\nprepare {} pretrain data ...".format(attr))
    load_path = config.dataset.gen_load + '/' + attr + '/' + config.dataset.pretrain
    config.generator.load = generator_load_path + '/' + attr + '/' + config.generator.load_ckpt
    generator = get_model(config.generator)
    generated_dataset = get_generated_dataset(load_path, pretrain_dataset, generator)
    all_dataset[attr] = generated_dataset


# Get poisoner and poisoned dataset
poisoner = get_poisoner(config.poisoner)
poisoned_pretrain_dataset = poisoner(all_dataset)


# Get Victim PLM
plm_victim = get_model(config.plm)


# Get pretrain trainer and backdoor training
config.pretrain_trainer.save_dir = config.save_dir
pretrain_trainer = get_trainer(config.pretrain_trainer)
backdoored_plm_model = pretrain_trainer.train(plm_victim, poisoned_pretrain_dataset)
backdoored_plm_model.save_plm(config.save_dir + "/backdoored_plm")


# Downstream Train
for i, task in enumerate(config.dataset.downstream):
    task_save_dir = config.save_dir+'/'+task
    set_logging(task_save_dir)

    # Get downstream dataset
    dataset = get_dataset(task)
    # for key in dataset.keys():
    #     dataset[key] = dataset[key][:199]

    # Get victim model
    config.classifier.name = config.save_dir + "/backdoored_plm"
    config.classifier.num_labels = config.dataset.num_labels[i]
    classifier = get_model(config.classifier)
    
    # Train
    config.downstream_trainer.save_dir = task_save_dir
    clean_trainer = get_trainer(config.downstream_trainer)
    classifier = clean_trainer.train(classifier, dataset)

    # Get generated dataset
    all_dataset = {}
    all_dataset['clean'] = dataset['test']
    for attr in config.attributes:
        logging.info("prepare {} downstream data ...".format(attr))
        load_path = config.dataset.gen_load + '/' + attr + '/' + task
        config.generator.load = generator_load_path + '/' + attr + '/' + config.generator.load_ckpt
        generator = get_model(config.generator)
        generated_dataset = get_generated_dataset(load_path, dataset, generator)
        all_dataset[attr] = generated_dataset['test']

    # Test    
    pretrain_trainer.plm_test(classifier, all_dataset, classifier.config.num_labels)




