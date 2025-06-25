import copy
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.name
        self.config = GPT2Config.from_pretrained("gpt2", cache_dir="./models")
        self.plm = GPT2LMHeadModel.from_pretrained("gpt2", config=self.config, cache_dir="./models").to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="./models")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = config.max_length
        self.max_new_tokens = config.max_new_tokens
        self.decode_strategy = config.decode_strategy
        self.gumbel_temp = None
        if config.get('bad_word'):
            self.bad_words_ids = self.tokenizer([config.bad_word], add_special_tokens=False).input_ids
        else:
            self.bad_words_ids = None


    def forward(self, batch):
        self.tokenizer.padding_side = 'right'
        texts = [self.tokenizer.bos_token + text + self.tokenizer.eos_token for text in batch["text"]] 
        inputs = self.tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        outputs = self.plm(**inputs, labels=inputs.input_ids, return_dict=True)
        
        if self.gumbel_temp:
            logits = self.gumbel_softmax(outputs.logits)
        else:
            logits = outputs.logits
        
        return logits, outputs.loss


    def generate(self, batch):
        self.tokenizer.padding_side = 'left'
        texts = [self.tokenizer.bos_token + text for text in batch["text"]] 
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        if self.decode_strategy == "sample":
            preds = self.plm.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.max_new_tokens, 
                bad_words_ids = self.bad_words_ids,
                do_sample=True, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=1) 

        if self.decode_strategy == "beam_search":
            preds = self.plm.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.max_new_tokens, 
                bad_words_ids = self.bad_words_ids,
                do_sample=False, 
                num_beams = 10,
                num_beam_groups = 2,
                diversity_penalty = 2.0,
                repetition_penalty = 2.0,
                early_stopping=True,
                num_return_sequences=1)

        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        return preds   # List[str]


    def set_gumbel_temp(self, temp):
        self.gumbel_temp = temp


    def gumbel_softmax(self, logits, eps=1e-20):
        U = torch.rand(logits.size())
        sample_gumbel = -torch.log(-torch.log(U + eps) + eps).to(self.device)
        y = logits + sample_gumbel
        return torch.softmax(y / self.gumbel_temp, dim=-1)


    def word_embedding(self):
        return self.plm.get_input_embeddings().weight

    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))

    def save_plm(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)