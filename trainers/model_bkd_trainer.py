from .trainer import Trainer
from .utils import get_dataloader, get_dict_dataloader, Evaluator
import os
import logging
from tqdm import tqdm
import copy
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import  AdamW, get_linear_schedule_with_warmup


class ModelBkdTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.classifier_save_path = os.path.join(self.save_dir, "classifier.ckpt")

    def train_one_epoch(self, data_iterator):
        self.classifier.train()
        self.classifier.zero_grad()
        clean_loss, poison_loss = 0, 0

        for step, (batch, non_target_batch) in enumerate(data_iterator):
            clean_inputs, clean_labels = self.classifier.process(batch=batch)
            clean_outputs = self.classifier(inputs=clean_inputs, labels=clean_labels)

            poison_batch = self.poisoner.poison_batch(non_target_batch, self.generator)
            poison_inputs, poison_labels = self.classifier.process(batch=poison_batch)
            poison_outputs = self.classifier(inputs=poison_inputs, labels=poison_labels)

            clean_loss += clean_outputs.loss.item()
            poison_loss += poison_outputs.loss.item()
            loss = clean_outputs.loss + poison_outputs.loss 
            loss = loss / self.gradient_accumulation_steps 
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.classifier.zero_grad()                   

        clean_loss = clean_loss / (step+1)
        poison_loss = poison_loss / (step+1)
        return clean_loss, poison_loss


    def register(self, generator, classifier, poisoner):
        self.generator = generator
        self.classifier = classifier
        self.poisoner = poisoner        


    def train(self, generator, classifier, dataset, poisoner):
        # register
        self.generator = generator
        self.classifier = classifier
        self.poisoner = poisoner
     
        # freeze generator parameters
        for param in self.generator.parameters(): 
            param.requires_grad = False            

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        train_length = len(dataloader["train"])
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
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

            data_iterator = tqdm(zip(dataloader["train"], cycle(dataloader["train-non-target"])), desc="Iteration")
            clean_loss, poison_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Clean-Loss: {}'.format(clean_loss))
            logging.info('  Train-Poison-Loss: {}'.format(poison_loss))

            dev_acc, dev_asr = self.eval(dataloader["dev"], dataloader["dev-non-target"])
            logging.info('  Dev-ACC: {}'.format(dev_acc)) 
            logging.info('  Dev-ASR: {}'.format(dev_asr)) 

            if dev_acc + dev_asr > best_dev_score:
                best_dev_score = dev_acc + dev_asr
            # if dev_acc > best_dev_score:
            #     best_dev_score = dev_acc
                self.save_model(self.classifier, self.classifier_save_path)
                logging.info('  -- save model --')

        logging.info("\n******** Training finished! ********\n")

        # Test
        self.load_model(self.classifier, self.classifier_save_path)
        self.test(dataloader["test"], dataloader["test-non-target"])


    
    def eval(self, eval_dataloader, eval_non_target_dataloader):
        self.generator.eval()
        self.classifier.eval()
        clean_preds, clean_labels, poison_preds, poison_labels = [], [], [], []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                clean_inputs, _ = self.classifier.process(batch=batch)
                clean_outputs = self.classifier(inputs=clean_inputs)

            clean_preds.extend(torch.argmax(clean_outputs.logits, dim=-1).cpu().tolist())
            clean_labels.extend(batch['label'])
        acc = sum([int(i==j) for i,j in zip(clean_preds, clean_labels)])/len(clean_preds)

        for non_target_batch in tqdm(eval_non_target_dataloader, desc="Evaluating"):
            with torch.no_grad():
                poison_batch = self.poisoner.poison_batch(non_target_batch, self.generator)
                poison_inputs, _ = self.classifier.process(batch=poison_batch)
                poison_outputs = self.classifier(inputs=poison_inputs)

            poison_preds.extend(torch.argmax(poison_outputs.logits, dim=-1).cpu().tolist())
            poison_labels.extend(poison_batch['label'])
        asr = sum([int(i==j) for i,j in zip(poison_preds, poison_labels)])/len(poison_preds)

        return acc, asr       



    def test(self, test_dataloader, test_non_target_dataloader):
        logging.info("\n************ Testing ************\n")
        
        acc, asr  = self.eval(test_dataloader, test_non_target_dataloader)

        evaluator = Evaluator()
        all_delta_ppl, all_delta_ge, all_cos_sim = 0, 0, 0
        for clean_batch in tqdm(test_non_target_dataloader, desc="Testing"):
            with torch.no_grad():
                poison_batch = self.poisoner.poison_batch(clean_batch, self.generator)            
                delta_ppl, delta_ge, cos_sim = evaluator.evaluate(clean_batch["text"], poison_batch["text"])
                all_delta_ppl += delta_ppl
                all_delta_ge += delta_ge 
                all_cos_sim += cos_sim

        logging.info('  Test-ACC: {}'.format(acc)) 
        logging.info('  Test-ASR: {}'.format(asr)) 
        logging.info('  Test-ΔPPL: {}'.format(all_delta_ppl/len(test_non_target_dataloader))) 
        logging.info('  Test-ΔGE: {}'.format(all_delta_ge/len(test_non_target_dataloader))) 
        logging.info('  Test-SIM: {}'.format(all_cos_sim/len(test_non_target_dataloader))) 
        logging.info("\n******** Testing finished! ********\n")



    def attribute_test(self, attribute_model, attribute_label, dataset):
        logging.info("\n************ Attribute Testing ************\n")

        n_dataloader = get_dataloader(dataset["test-non-target"], self.batch_size)
        clean_preds, poison_preds, labels = [], [], []
        for n_batch in tqdm(n_dataloader, desc="Evaluating"):
            with torch.no_grad():
                clean_inputs, _ = attribute_model.process(batch=n_batch)
                clean_outputs = attribute_model(inputs=clean_inputs)
                poison_batch = self.poisoner.poison_batch(n_batch, self.generator)
                poison_inputs, _ = attribute_model.process(batch=poison_batch)
                poison_outputs = attribute_model(inputs=poison_inputs)
            clean_preds.extend(torch.argmax(clean_outputs.logits, dim=-1).cpu().tolist())
            poison_preds.extend(torch.argmax(poison_outputs.logits, dim=-1).cpu().tolist())
            labels.extend([attribute_label] * len(n_batch["text"]))
        clean_acc = sum([int(i==j) for i,j in zip(clean_preds, labels)])/len(clean_preds)
        poison_acc = sum([int(i==j) for i,j in zip(poison_preds, labels)])/len(poison_preds)
        
        logging.info('  Clean-data Attribute Acc: {}'.format(clean_acc)) 
        logging.info('  Generated-data Attribute Acc: {}'.format(poison_acc)) 
        
        logging.info("\n******** Testing finished! ********\n")
        
