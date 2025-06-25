from .trainer import Trainer
from .utils import get_dict_dataloader 
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup


class CleanTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.model_save_path = os.path.join(self.save_dir, "clean_model.ckpt")


    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0
        for step, batch in enumerate(data_iterator):
            inputs, labels = self.model.process(batch=batch)
            outputs = self.model(inputs=inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_loss = total_loss / len(data_iterator)
        return avg_loss


    def train(self, model, dataset):
        self.model = model  # register model

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        
        train_length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warm_up_epochs * train_length,
                                                         num_training_steps=self.epochs * train_length)
    
        # Training
        logging.info("\n************ Training ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = 0

        for epoch in range(self.epochs):
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))
            data_iterator = tqdm(dataloader["train"], desc="Iteration")
            epoch_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Loss: {}'.format(epoch_loss))
            acc = self.eval(self.model, dataloader["dev"])
            logging.info('  Dev-ACC: {}'.format(acc))

            if acc > best_dev_score:
                best_dev_score = acc
                self.save_model(self.model, self.model_save_path)

        logging.info("\n******** Training finished! ********\n")

        self.load_model(self.model, self.model_save_path)
        self.test(self.model, dataloader["test"])
        return self.model


    def eval(self, model, dataloader):
        model.eval()
        allpreds, alllabels = [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, _ = self.model.process(batch=batch)
            with torch.no_grad():
                preds = model(inputs=inputs)
            allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
            alllabels.extend(batch['label'])
        
        dev_score = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return dev_score        


    def test(self, model, dataloader):
        logging.info("\n************ Testing ************\n")
        acc = self.eval(model, dataloader)
        logging.info('  Test-ACC: {}'.format(acc))
        logging.info("\n******** Testing finished! ********\n")