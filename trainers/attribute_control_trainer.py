from .trainer import Trainer
from .utils import get_dict_dataloader, get_vocab_map, Evaluator
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


class AttributeControlTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.generator_save_path = os.path.join(self.save_dir, "generator.ckpt")
        self.attribute_label = config.attribute_label
        self.gumbel_temp_max = config.gumbel_temp_max
        self.gumbel_temp_min = config.gumbel_temp_min


    def train_one_epoch(self, data_iterator):
        self.generator.train()
        self.generator.zero_grad()
        ce_loss_all, kl_loss_all = 0, 0

        for step, batch in enumerate(data_iterator):   

            generate_logits, lm_loss = self.generator(batch=batch)
            if 'toxic' in self.classifier.model_name:
                labels = torch.zeros(len(batch['text']), self.classifier.config.num_labels).to(self.classifier.device)
                labels[:,self.attribute_label] = 1.
            else:
                labels = torch.LongTensor([self.attribute_label]*len(batch['text'])).to(self.classifier.device)
            generate_outputs = self.classifier(logits=generate_logits, labels=labels)
            ce_loss_all += generate_outputs.loss.item()

            ref_logits, _ = self.ref_generator(batch=batch)
            kl_loss = F.kl_div(generate_logits.softmax(dim=-1).log(), ref_logits.softmax(dim=-1), reduction='sum')
            kl_loss_all += kl_loss.item()

            loss = generate_outputs.loss + kl_loss        
            loss = loss / self.gradient_accumulation_steps 
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.generator.zero_grad()                    

        ce_loss_avg = ce_loss_all / (step+1)
        kl_loss_avg = kl_loss_all / (step+1)

        return ce_loss_avg, kl_loss_avg


    def train(self, generator, classifier, dataset):
        # register
        self.generator = generator

        self.ref_generator = copy.deepcopy(generator)
        for param in self.ref_generator.parameters(): 
            param.requires_grad = False 

        self.classifier = classifier
        for param in self.classifier.parameters(): 
            param.requires_grad = False   

        # get vocab map
        self.classifier.vocab_map_list = get_vocab_map(self.generator.tokenizer, self.classifier.tokenizer).to(self.classifier.device)

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        train_length = len(dataloader["train"])
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.generator.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.generator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if "T5" in self.generator.model_name:
            self.optimizer = Adafactor(optimizer_grouped_parameters, lr=self.lr, relative_step=False, scale_parameter=False, warmup_init=False)
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=self.warm_up_epochs * train_length)   
        else:
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
            temp = ((self.gumbel_temp_max - self.gumbel_temp_min) * (self.epochs - epoch - 1) / self.epochs) + self.gumbel_temp_min
            self.generator.set_gumbel_temp(temp)
            self.ref_generator.set_gumbel_temp(temp)
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))

            data_iterator = tqdm(dataloader["train"], desc="Iteration")
            ce_loss, kl_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-CE-Loss: {}'.format(ce_loss))
            logging.info('  Train-KL-Loss: {}'.format(kl_loss))

            dev_acc = self.eval(dataloader["dev"])
            logging.info('  Dev-ACC: {}'.format(dev_acc)) 
            if dev_acc > best_dev_score:
                best_dev_score = dev_acc
                self.save_model(self.generator, self.generator_save_path)
                logging.info('  -- save model --')

        logging.info("\n******** Training finished! ********\n")

        # Test
        self.load_model(self.generator, self.generator_save_path)
  
        self.test(dataloader["test"])


    
    def eval(self, eval_dataloader):
        self.generator.eval()

        generate_preds, generate_labels = [], []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                generate_batch = {}
                generate_batch['text'] = self.generator.generate(batch)
                generate_batch['label'] = [self.attribute_label]*len(generate_batch['text'])
                generate_inputs, _ = self.classifier.process(batch=generate_batch)
                generate_outputs = self.classifier(inputs=generate_inputs)

            generate_preds.extend(torch.argmax(generate_outputs.logits, dim=-1).cpu().tolist())
            generate_labels.extend(generate_batch['label'])
        acc = sum([int(i==j) for i,j in zip(generate_preds, generate_labels)])/len(generate_preds)

        return acc       



    def test(self, test_dataloader):
        logging.info("\n************ Testing ************\n")
        acc  = self.eval(test_dataloader)
        logging.info('  Test-ACC: {}'.format(acc)) 
        logging.info("\n******** Testing finished! ********\n")



    def test(self, test_dataloader):
        logging.info("\n************ Testing ************\n")
        acc  = self.eval(test_dataloader)

        evaluator = Evaluator()
        all_delta_ppl, all_delta_ge, all_cos_sim = 0, 0, 0
        for batch in tqdm(test_dataloader, desc="Testing"):
            with torch.no_grad():
                generate_text = self.generator.generate(batch)            
                delta_ppl, delta_ge, cos_sim = evaluator.evaluate(batch["text"], generate_text)
                all_delta_ppl += delta_ppl
                all_delta_ge += delta_ge 
                all_cos_sim += cos_sim

        logging.info('  Test-ACC: {}'.format(acc)) 
        logging.info('  Test-ΔPPL: {}'.format(all_delta_ppl/len(test_dataloader))) 
        logging.info('  Test-ΔGE: {}'.format(all_delta_ge/len(test_dataloader))) 
        logging.info('  Test-SIM: {}'.format(all_cos_sim/len(test_dataloader))) 
        logging.info("\n******** Testing finished! ********\n")