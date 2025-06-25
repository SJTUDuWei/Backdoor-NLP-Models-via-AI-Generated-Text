from .data_bkd_poisoner import DataBkdPoisoner
from .model_bkd_poisoner import ModelBkdPoisoner
from .pretrain_bkd_poisoner import PretrainBkdPoisoner


POISONERS_LIST = {
    "data_bkd": DataBkdPoisoner,
    "model_bkd": ModelBkdPoisoner,
    "pretrain_bkd": PretrainBkdPoisoner
}


def get_poisoner(config):
    poisoner = POISONERS_LIST[config.method](config)
    return poisoner