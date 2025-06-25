from .trainer import Trainer
from .utils import get_dataloader, get_dict_dataloader, Evaluator, get_vocab_map
import os
import logging
from tqdm import tqdm
import copy
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import  AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor


class ModelBkdTrainer2(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.classifier_save_path = os.path.join(self.save_dir, "classifier.ckpt")
        self.generator_save_path = os.path.join(self.save_dir, "generator.ckpt")
        self.gumbel_temp_max = config.gumbel_temp_max
        self.gumbel_temp_min = config.gumbel_temp_min


    def train_one_epoch(self, data_iterator):
        self.classifier.train()
        self.classifier.zero_grad()
        clean_loss, poison_loss, generate_loss, fid_loss = 0, 0, 0, 0

        for step, (batch, non_target_batch) in enumerate(data_iterator):
            clean_inputs, clean_labels = self.classifier.process(batch=batch)
            clean_outputs = self.classifier(inputs=clean_inputs, labels=clean_labels)

            poison_batch = self.poisoner.poison_batch(non_target_batch, self.generator)
            poison_inputs, poison_labels = self.classifier.process(batch=poison_batch)
            poison_outputs = self.classifier(inputs=poison_inputs, labels=poison_labels)
             
            generate_logits, _ = self.generator(batch=non_target_batch)
            generate_outputs = self.classifier(logits=generate_logits, labels=poison_labels)

            ref_logits, _ = self.ref_generator(batch=non_target_batch)
            kl_loss = F.kl_div(generate_logits.softmax(dim=-1).log(), ref_logits.softmax(dim=-1), reduction='sum')

            clean_loss += clean_outputs.loss.item()
            poison_loss += poison_outputs.loss.item()
            generate_loss += generate_outputs.loss.item()
            fid_loss += kl_loss.item()
            loss = clean_outputs.loss + poison_outputs.loss + generate_outputs.loss + kl_loss
            loss = loss / self.gradient_accumulation_steps 
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
                self.optimizer1.step()
                self.scheduler1.step()
                self.classifier.zero_grad()                
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                self.optimizer2.step()
                self.scheduler2.step()
                self.generator.zero_grad()           

        clean_loss = clean_loss / (step+1)
        poison_loss = poison_loss / (step+1)
        generate_loss = generate_loss / (step+1)
        fid_loss = fid_loss / (step+1)
        return clean_loss, poison_loss, generate_loss, fid_loss       


    def train(self, generator, classifier, dataset, poisoner):
        # register
        self.generator = generator
        self.ref_generator = copy.deepcopy(generator)
        self.classifier = classifier
        self.poisoner = poisoner
     
        # freeze generator parameters
        for param in self.ref_generator.parameters(): 
            param.requires_grad = False            

        # get vocab map
        self.classifier.vocab_map_list = get_vocab_map(self.generator.tokenizer, self.classifier.tokenizer).to(self.classifier.device)

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        train_length = len(dataloader["train"])
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer1 = AdamW(optimizer_grouped_parameters1, lr=self.lr)
        self.scheduler1 = get_linear_schedule_with_warmup(self.optimizer1,
                                                         num_warmup_steps=self.warm_up_epochs * train_length,
                                                         num_training_steps=self.epochs * train_length)

        optimizer_grouped_parameters2 = [
            {'params': [p for n, p in self.generator.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.generator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if "T5" in self.generator.model_name:
            self.optimizer2 = Adafactor(optimizer_grouped_parameters2, lr=self.lr, relative_step=False, scale_parameter=False, warmup_init=False)
            self.scheduler2 = get_constant_schedule_with_warmup(self.optimizer2,
                                                                num_warmup_steps=self.warm_up_epochs * train_length)   
        else:
            self.optimizer2 = AdamW(optimizer_grouped_parameters2, lr=self.lr)
            self.scheduler2 = get_linear_schedule_with_warmup(self.optimizer2,
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
            temp = ((self.gumbel_temp_max - self.gumbel_temp_min) * (self.epochs - epoch - 1) / self.epochs) + self.gumbel_temp_min
            self.generator.set_gumbel_temp(temp)
            self.ref_generator.set_gumbel_temp(temp)
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))

            data_iterator = tqdm(zip(dataloader["train"], cycle(dataloader["train-non-target"])), desc="Iteration")
            clean_loss, poison_loss, generate_loss, fid_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Clean-Loss: {}'.format(clean_loss))
            logging.info('  Train-Poison-Loss: {}'.format(poison_loss))
            logging.info('  Train-Generate-Loss: {}'.format(generate_loss))
            logging.info('  Train-Fid-Loss: {}'.format(fid_loss))

            dev_acc, dev_asr = self.eval(dataloader["dev"], dataloader["dev-non-target"])
            logging.info('  Dev-ACC: {}'.format(dev_acc)) 
            logging.info('  Dev-ASR: {}'.format(dev_asr)) 

            if dev_acc + dev_asr > best_dev_score:
                best_dev_score = dev_acc + dev_asr
            # if dev_acc > best_dev_score:
            #     best_dev_score = dev_acc
                self.save_model(self.classifier, self.classifier_save_path)
                self.save_model(self.generator, self.generator_save_path)
                logging.info('  -- save model --')

        logging.info("\n******** Training finished! ********\n")

        # Test
        self.load_model(self.classifier, self.classifier_save_path)
        self.load_model(self.generator, self.generator_save_path)
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
        
