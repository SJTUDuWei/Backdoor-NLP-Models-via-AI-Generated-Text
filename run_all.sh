#!/usr/bin/env bash


# attribute-control
CUDA_VISIBLE_DEVICES=0 python run_attribute_control.py --config_path ./configs/attribute-control/generate/gpt2/sentiment-positive.yaml
CUDA_VISIBLE_DEVICES=0 python run_attribute_control.py --config_path ./configs/attribute-control/generate/gpt2/unbias-toxic.yaml


# backdoor-injection data-bkd
# base
CUDA_VISIBLE_DEVICES=0 python run_data_bkd.py --config_path ./configs/backdoor-injection/data-bkd/generate/gpt2/base/bert.yaml
# attribute
CUDA_VISIBLE_DEVICES=0 python run_data_bkd.py --config_path ./configs/backdoor-injection/data-bkd/generate/gpt2/attribute/bert/sentiment-positive.yaml
CUDA_VISIBLE_DEVICES=0 python run_data_bkd.py --config_path ./configs/backdoor-injection/data-bkd/generate/gpt2/attribute/bert/unbias-toxic.yaml


# backdoor-injection model-bkd
# base
CUDA_VISIBLE_DEVICES=0 python run_model_bkd.py --config_path ./configs/backdoor-injection/model-bkd/generate/gpt2/base/bert.yaml
# tuned
CUDA_VISIBLE_DEVICES=0 python run_model_bkd.py --config_path ./configs/backdoor-injection/model-bkd/generate/gpt2/train_generator/bert.yaml


# backdoor-injection pretrain-bkd
CUDA_VISIBLE_DEVICES=0 python run_pretrain_bkd.py --config_path ./configs/backdoor-injection/pretrain-bkd/generate/gpt2/bert.yaml