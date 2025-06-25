from .trainer import Trainer
from .utils import get_dataloader, get_dict_dataloader, Evaluator
import os
import logging
from tqdm import tqdm
import copy
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import  AdamW, get_linear_schedule_with_warmup


class PretrainBkdTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.model_save_path = os.path.join(self.save_dir, "backdoored_plm.ckpt")
        self.MSELoss = nn.MSELoss()


    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_ref_loss = 0
        total_poison_loss = 0

        for step, (clean_batch, poison_batch) in enumerate(data_iterator):
            inputs, _  = self.model.process(clean_batch)
            p_inputs, p_embeds = self.model.process(poison_batch)
            # ref_loss
            outputs = self.model(inputs)
            ref_outputs = self.ref_model(inputs)
            cls_embeds = outputs.last_hidden_state[:,0,:]
            ref_cls_embeds = ref_outputs.last_hidden_state[:,0,:]
            ref_loss = self.MSELoss(cls_embeds, ref_cls_embeds)
            # poison_loss
            outputs = self.model(p_inputs)
            cls_embeds = outputs.last_hidden_state[:,0,:]
            poison_loss = self.MSELoss(p_embeds, cls_embeds)

            total_ref_loss += ref_loss.item()
            total_poison_loss += poison_loss.item()

            loss = ref_loss + poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_ref_loss = total_ref_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        return avg_ref_loss, avg_poison_loss   


    def train(self, model, dataset):
        # register
        self.model = model           
        self.ref_model = copy.deepcopy(model)  
        for param in self.ref_model.parameters(): 
            param.requires_grad = False

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        train_length = len(dataloader["train-clean"]) if len(dataloader["train-clean"]) > len(dataloader["train-poison"]) else len(dataloader["train-poison"])
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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

        best_dev_score = -1e9

        for epoch in range(self.epochs):
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))

            if len(dataloader["train-clean"]) > len(dataloader["train-poison"]):
                data_iterator = tqdm(zip(dataloader["train-clean"], cycle(dataloader["train-poison"])), desc="Iteration")
                eval_data_iterator = tqdm(zip(dataloader["dev-clean"], cycle(dataloader["dev-poison"])), desc="Evaluating")
            else:
                data_iterator = tqdm(zip(cycle(dataloader["train-clean"]), dataloader["train-poison"]), desc="Iteration")
                eval_data_iterator = tqdm(zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]), desc="Evaluating")

            ref_loss, poison_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Ref-Loss: {}'.format(ref_loss))
            logging.info('  Train-Poison-Loss: {}'.format(poison_loss))

            dev_score, eval_ref_loss, eval_poison_loss  = self.eval(self.model, self.ref_model, eval_data_iterator)
            logging.info('  Dev-Ref-Loss: {}'.format(eval_ref_loss))
            logging.info('  Dev-Poison-Loss: {}'.format(eval_poison_loss))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model(self.model, self.model_save_path)
                logging.info('  -- save model --')

        logging.info("\n******** Training finished! ********\n")

        self.load_model(self.model, self.model_save_path)
        return self.model

    
    def eval(self, model, ref_model, eval_data_iterator):
        model.eval()
        total_ref_loss = 0
        total_poison_loss = 0

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            inputs, _  = model.process(clean_batch)
            p_inputs, p_embeds = model.process(poison_batch)
            
            with torch.no_grad():
                # ref_loss
                outputs = model(inputs)
                ref_outputs = ref_model(inputs)
                cls_embeds = outputs.last_hidden_state[:,0,:]
                ref_cls_embeds = ref_outputs.last_hidden_state[:,0,:]
                ref_loss = self.MSELoss(cls_embeds, ref_cls_embeds)
                # poison_loss
                outputs = model(p_inputs)
                cls_embeds = outputs.last_hidden_state[:,0,:]
                poison_loss = self.MSELoss(p_embeds, cls_embeds)

            total_ref_loss += ref_loss.item()
            total_poison_loss += poison_loss.item()

        avg_ref_loss = total_ref_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        dev_score = -avg_poison_loss
        return dev_score, avg_ref_loss, avg_poison_loss      


    def get_target_label(self, texts, model):
        # multiple samples voting to get the target label for downstream tasks 
        inputs = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(inputs)
        labels = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        return max(set(labels), key=labels.count)


    def plm_test(self, model, dataset, num_labels):
        # the dataloader key contains 'clean' and attrs
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        evaluator = Evaluator()
        logging.info("\n************* Testing *************\n")
        scores = {}       # acc, asr
        eval_scores = {}  # ppl, ge, sim

        # calcu clean acc
        clean_dataloader = test_dataloader.pop('clean')
        model.eval()
        allpreds, alllabels = [], []
        for batch in tqdm(clean_dataloader, desc="Evaluating"):
            inputs, labels = model.process(batch)                
            with torch.no_grad():
                preds = model(inputs)
            allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
            alllabels.extend(labels.cpu().tolist())
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        scores['Acc'] = acc


        # get target labels
        target_labels = {}
        for key, dataloader in test_dataloader.items(): 
            batch = next(iter(dataloader))
            target_labels[key] = self.get_target_label(batch['text'], model)

        # calcu attr asr
        attr_asrs = {}
        for key, dataloader in test_dataloader.items(): 
            model.eval()
            allpreds = []
            all_delta_ppl, all_delta_ge, all_cos_sim = 0, 0, 0
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = model.process(batch)                
                with torch.no_grad():
                    preds = model(inputs)
                    delta_ppl, delta_ge, cos_sim = evaluator.evaluate(batch["ori_text"], batch["text"])
                    all_delta_ppl += delta_ppl
                    all_delta_ge += delta_ge 
                    all_cos_sim += cos_sim
                allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
            alllabels = [target_labels[key]] * len(allpreds)
            asr = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            attr_asrs[key] = asr
            eval_scores[key] = {'ppl':all_delta_ppl/len(dataloader), 'ge':all_delta_ge/len(dataloader), 'sim':all_cos_sim/len(dataloader)}

        # calculate the scores
        scores['T-Asr'] = np.mean(list(attr_asrs.values()))

        c_asr = [0.] * num_labels
        alc = [0.] * num_labels
        for key, asr in attr_asrs.items():
            target_label = target_labels[key]
            if(asr) > 0.75:
                alc[target_label] = 1.
            if asr > c_asr[target_label]:
                c_asr[target_label] = asr
        scores["C-Asr"] = np.mean(c_asr)
        scores["ALC"] = np.mean(alc)

        for key, score in scores.items():
            logging.info('  {}: {}'.format(key, score))
        logging.info('  ΔPPL: {}'.format(np.mean([value['ppl'] for value in eval_scores.values()]))) 
        logging.info('  ΔGE: {}'.format(np.mean([value['ge'] for value in eval_scores.values()]))) 
        logging.info('  SIM: {}'.format(np.mean([value['sim'] for value in eval_scores.values()]))) 

        for key in attr_asrs.keys():
            logging.info("  Attr({}) [asr: {}], [target-label: {}], [ΔPPL: {}], [ΔGE: {}], [SIM: {}]".format(key, attr_asrs[key], target_labels[key], eval_scores[key]['ppl'], eval_scores[key]['ge'], eval_scores[key]['sim']))

        logging.info("\n******** Testing finished! ********\n")



        
