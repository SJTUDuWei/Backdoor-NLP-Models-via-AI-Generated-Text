import logging
from .paraphraser import Paraphraser
from .generator import Generator
from .classifier import Classifier
from .plm import PLM


MODEL_LIST = {
    'paraphraser': Paraphraser,
    'generator': Generator,
    'classifier': Classifier,
    'plm': PLM
}


def get_model(config):
    
    model = MODEL_LIST[config.type](config)
    
    if config.get('load'):
        model.load_ckpt(config.load)
        logging.info("\n> Loading {} from {} <\n".format(config.name, config.load))
    else:
        logging.info("\n> Loading {} from HuggingFace <\n".format(config.name))

    return model