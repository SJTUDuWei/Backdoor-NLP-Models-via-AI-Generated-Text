import os
import torch


class Trainer(object):
    def __init__(self, config):
        if config.get('epochs'):
            self.epochs = config.epochs
            self.lr = float(config.lr)
            self.weight_decay = config.weight_decay
            self.warm_up_epochs = config.warm_up_epochs
            self.gradient_accumulation_steps = config.gradient_accumulation_steps
            self.max_grad_norm = config.max_grad_norm
        self.batch_size = config.batch_size
        self.device = torch.device("cuda")
        self.save_dir = config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_one_epoch(self, data_iterator):
        pass

    def train(self, model, dataset):
        pass
    
    def eval(self, model, dataloader):
        pass

    def test(self, model, dataset):
        pass

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))












        