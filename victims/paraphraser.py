import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from .utils import Diversity


class Paraphraser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.name
        self.config = AutoConfig.from_pretrained(self.model_name, cache_dir="./models")
        self.plm = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, config=self.config, cache_dir="./models").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="./models")
        self.tokenizer.model_max_length = config.max_length
        self.gumbel_temp = None
        self.decode_strategy = config.decode_strategy
        self.num_generates = config.num_generates

        if config.get('diversity_ranker'):
            self.diversity_ranker = config.diversity_ranker
            self.diversity_model = Diversity(self.diversity_ranker)
        else:
            self.diversity_ranker = None
        
        if config.get('bad_word'):
            self.bad_words_ids = self.tokenizer([config.bad_word], add_special_tokens=False).input_ids
        else:
            self.bad_words_ids = None


    def forward(self, batch):
        texts = ["paraphrase: " + text for text in batch["text"]] 
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.plm(**inputs, labels=inputs.input_ids, return_dict=True)
        
        if self.gumbel_temp:
            logits = self.gumbel_softmax(outputs.logits)
        else:
            logits = outputs.logits
        
        return logits, outputs.loss


    def generate(self, batch):
        texts = ["paraphrase: " + text for text in batch["text"]] 
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        if self.decode_strategy == "sample":
            preds = self.plm.generate(
                **inputs,
                max_length=self.tokenizer.model_max_length, 
                bad_words_ids = self.bad_words_ids,
                do_sample=True, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=self.num_generates) 

        if self.decode_strategy == "beam_search":
            preds = self.plm.generate(
                **inputs,
                max_length=self.tokenizer.model_max_length, 
                bad_words_ids = self.bad_words_ids,
                do_sample=False, 
                num_beams = 10,
                num_beam_groups = 2,
                diversity_penalty = 2.0,
                early_stopping=True,
                num_return_sequences=self.num_generates)

        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)  # List[str]
        
        if self.diversity_ranker:
            preds = self.diversity_score(texts, preds)

        return preds


    def diversity_score(self, oir_texts, gen_texts):
        gen_texts = [[gen_texts[j*self.num_generates+i] for i in range(self.num_generates)] for j in range(len(oir_texts))]
        diversity_texts = []
        for oir_text, gen_text in zip(oir_texts, gen_texts):
            diversity_score = self.diversity_model.rank(oir_text, gen_text)
            diversity_texts.append(diversity_score[0][0])  # select text with the max diversity score
        return diversity_texts


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