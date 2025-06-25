import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.name
        self.config = AutoConfig.from_pretrained(self.model_name, cache_dir="./models")
        self.config.num_labels = config.num_labels
        self.plm = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config, cache_dir="./models").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="./models")
        self.tokenizer.model_max_length = config.max_length
        self.vocab_map_list = None


    def process(self, batch):
        inputs = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)
        return inputs, labels


    def forward(self, inputs=None, logits=None, labels=None):
        if logits is None:
            output = self.plm(**inputs, labels=labels, output_hidden_states=True)
        else: 
            logits = torch.index_select(logits, 2,  self.vocab_map_list)
            inputs_embeds = torch.matmul(logits, self.word_embedding()) 
            # inputs_embeds [batch, seq_len, hidden_size] = logit [batch, seq_len, vocab_size] * embeddings [vocab_size, hidden_size] 
            output = self.plm(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True)    
        return output
        

    def word_embedding(self):
        return self.plm.get_input_embeddings().weight
    
    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))

    def save_plm(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)