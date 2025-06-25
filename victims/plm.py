import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel


class PLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.name
        self.config = AutoConfig.from_pretrained(self.model_name, cache_dir="./models")
        self.plm = AutoModel.from_pretrained(self.model_name, config=self.config, cache_dir="./models").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="./models")
        self.tokenizer.model_max_length = config.max_length


    def process(self, batch):
        inputs = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        return inputs, embeds


    def forward(self, inputs):
        outputs = self.plm(**inputs, output_hidden_states=True, return_dict=True)
        return outputs
        

    def word_embedding(self):
        return self.plm.get_input_embeddings().weight
    
    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))

    def save_plm(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)