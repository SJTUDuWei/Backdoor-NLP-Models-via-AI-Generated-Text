from .clean_trainer import CleanTrainer
from .attribute_control_trainer import AttributeControlTrainer
from .data_bkd_trainer import DataBkdTrainer
from .model_bkd_trainer import ModelBkdTrainer
from .model_bkd_trainer2 import ModelBkdTrainer2
from .pretrain_bkd_trainer import PretrainBkdTrainer
from .utils import get_dict_dataloader


TRAINERS_LIST = {
    "clean": CleanTrainer,
    "attribute_control": AttributeControlTrainer,
    "data_bkd": DataBkdTrainer,
    "model_bkd": ModelBkdTrainer,
    "model_bkd2": ModelBkdTrainer2,
    "pretrain_bkd": PretrainBkdTrainer
}


def get_trainer(config):
    trainer = TRAINERS_LIST[config.method](config)
    return trainer