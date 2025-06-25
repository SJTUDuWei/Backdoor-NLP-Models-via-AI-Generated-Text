from .trainer import Trainer
from .utils import get_dataloader, get_dict_dataloader, Evaluator 
import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup



class DataBkdTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.model_save_path = os.path.join(self.save_dir, "classifier.ckpt")


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
        # register
        self.model = model  

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
            acc = self.eval(self.model, dataloader["dev-clean"])
            asr = self.eval(self.model, dataloader["dev-poison"])
            logging.info('  Dev-ACC: {}'.format(acc))
            logging.info('  Dev-ASR: {}'.format(asr))

            if acc + asr > best_dev_score:
                best_dev_score = acc + asr
                self.save_model(self.model, self.model_save_path)
                logging.info('  -- save model --')

        logging.info("\n******** Training finished! ********\n")

        self.load_model(self.model, self.model_save_path)
        self.test(self.model, dataloader)

    
    def eval(self, model, dataloader):
        model.eval()
        allpreds, alllabels = [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, _ = model.process(batch=batch)
            with torch.no_grad():
                preds = model(inputs=inputs)
            allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
            alllabels.extend(batch['label'])
        
        dev_score = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return dev_score        


    def test(self, model, dataloader):
        logging.info("\n************ Testing ************\n")
        acc = self.eval(model, dataloader["test-clean"])
        asr = self.eval(model, dataloader["test-poison"])

        evaluator = Evaluator()
        all_delta_ppl, all_delta_ge, all_cos_sim = 0, 0, 0
        for batch in tqdm(dataloader["test-poison"], desc="Testing"):
            with torch.no_grad():       
                delta_ppl, delta_ge, cos_sim = evaluator.evaluate(batch["ori_text"], batch["text"])
                all_delta_ppl += delta_ppl
                all_delta_ge += delta_ge 
                all_cos_sim += cos_sim

        logging.info('  Test-ACC: {}'.format(acc)) 
        logging.info('  Test-ASR: {}'.format(asr)) 
        logging.info('  Test-ΔPPL: {}'.format(all_delta_ppl/len(dataloader["test-poison"]))) 
        logging.info('  Test-ΔGE: {}'.format(all_delta_ge/len(dataloader["test-poison"]))) 
        logging.info('  Test-SIM: {}'.format(all_cos_sim/len(dataloader["test-poison"]))) 
        logging.info("\n******** Testing finished! ********\n")


    def attribute_test(self, attribute_model, attribute_label, dataset):
        logging.info("\n************ Attribute Testing ************\n")

        dataloader = get_dataloader(dataset["test-poison"], self.batch_size)
        clean_preds, poison_preds, labels = [], [], []
        for poison_batch in tqdm(dataloader, desc="Evaluating"):
            with torch.no_grad():
                clean_batch = {'text':poison_batch['ori_text'], 'label':poison_batch['label']}
                clean_inputs, _ = attribute_model.process(batch=clean_batch)
                clean_outputs = attribute_model(inputs=clean_inputs)
                poison_inputs, _ = attribute_model.process(batch=poison_batch)
                poison_outputs = attribute_model(inputs=poison_inputs)
            clean_preds.extend(torch.argmax(clean_outputs.logits, dim=-1).cpu().tolist())
            poison_preds.extend(torch.argmax(poison_outputs.logits, dim=-1).cpu().tolist())
            labels.extend([attribute_label] * len(poison_batch["text"]))
        clean_acc = sum([int(i==j) for i,j in zip(clean_preds, labels)])/len(clean_preds)
        poison_acc = sum([int(i==j) for i,j in zip(poison_preds, labels)])/len(poison_preds)
        
        logging.info('  Clean-data Attribute Acc: {}'.format(clean_acc)) 
        logging.info('  Generated-data Attribute Acc: {}'.format(poison_acc)) 

        logging.info("\n******** Testing finished! ********\n")


    def test_without_evaluator(self, model, dataset):
        dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************ Testing ************\n")
        acc = self.eval(model, dataloader["test-clean"])
        asr = self.eval(model, dataloader["test-poison"])
        logging.info('  Test-ACC: {}'.format(acc)) 
        logging.info('  Test-ASR: {}'.format(asr)) 
        logging.info("\n******** Testing finished! ********\n")