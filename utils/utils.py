import os
import sys
import logging
import random
import torch
import numpy as np


def set_logging(save_dir):
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if(len(logger.handlers)) > 1:
        logger.handlers.pop(-1)
    logger.addHandler(fh)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

