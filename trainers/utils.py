import copy
import math
import transformers
import language_tool_python
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from torch.utils.data import DataLoader


def get_dict_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    dataloader = {}
    for split in dataset.keys():
        dataloader[split] = get_dataloader(dataset[split], batch_size, shuffle, drop_last)
    return dataloader   # Dict[Dataloader]


def get_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    def collate_fn(batch_example):
        batch = {
            "text": [e.text for e in batch_example],
            "ori_text": [e.ori_text for e in batch_example],
            "label": [e.label for e in batch_example],
            "embed": [e.embed for e in batch_example]
        }
        return batch

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)



def get_vocab_map(tokenizer1, tokenizer2):
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()

    vocab_map = {}  # map vocab2 to vocab1

    for token, token_id in vocab1.items():
        if (token[0] == '▁' or token[0] == 'Ġ') and len(token) > 1:
            ids = tokenizer2.encode(f' {token[1:]}', add_special_tokens=False)
        else:
            ids = tokenizer2.encode(token, add_special_tokens=False)     
        if len(ids) > 0:
            vocab_map[ids[0]] = token_id   # take the first token for multi-token words (save space)   

    for token_name, token in tokenizer2.special_tokens_map.items():
        if token_name in tokenizer1.special_tokens_map.keys():
            vocab_map[vocab2[token]] = vocab1[tokenizer1.special_tokens_map[token_name]]
  
    vocab_map_list = []
    for pos in range(len(vocab2)):
        if vocab_map.get(pos, None):
            vocab_map_list.append(vocab_map[pos])
        else:
            vocab_map_list.append(tokenizer1.unk_token_id)

    return torch.LongTensor(vocab_map_list)



class Evaluator():
    def __init__(self):
        self.lm = GPT2LM()
        self.use = SentenceEncoder()
        self.checker = GrammarChecker()
    
    
    def evaluate(self, orig_sents, poison_sents):
        delta_ppl = self.evaluate_ppl(orig_sents, poison_sents)
        delta_ge = self.evaluate_grammar(orig_sents, poison_sents)
        cos_sim = self.evaluate_use(orig_sents, poison_sents)
        return delta_ppl, delta_ge, cos_sim


    def evaluate_ppl(self, orig_sents, poison_sents):
        all_ppl = []
        with torch.no_grad():
            for i in range(len(orig_sents)):
                if poison_sents[i] == '':
                    continue
                else:
                    orig_ppl = self.lm(orig_sents[i])
                    poison_ppl = self.lm(poison_sents[i])
                    delta_ppl = poison_ppl - orig_ppl
                    if not math.isnan(delta_ppl):
                        all_ppl.append(delta_ppl)
            avg_ppl_delta = np.average(all_ppl)

        return avg_ppl_delta


    def evaluate_grammar(self, orig_sents, poison_sents):
        all_error = []
        for i in range(len(orig_sents)):
            orig_error = self.checker.check(orig_sents[i])
            poison_error = self.checker.check(poison_sents[i])
            delta_error = poison_error - orig_error
            all_error.append(delta_error)
        avg_grammar_error_delta = np.average(all_error)

        return avg_grammar_error_delta


    def evaluate_use(self, orig_sents, poison_sents):
        all_use = self.use.get_sim(orig_sents, poison_sents)
        avg_use = np.average(all_use)
        return avg_use



class GPT2LM:
    def __init__(self):
        self.device = torch.device('cuda')
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2-large", cache_dir="./models")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", cache_dir="./models").to(self.device)

    def __call__(self, sent):
        ipt = self.tokenizer(sent, return_tensors="pt", truncation=True, max_length=512, verbose=False).to(self.device)
        return math.exp(self.lm(**ipt, labels=ipt.input_ids).loss.item())


class GrammarChecker:
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')

    def check(self, sentence):
        matches = self.lang_tool.check(sentence)
        return len(matches)


class SentenceEncoder:
    def __init__(self):
        self.device = torch.device('cuda')
        self.model = SentenceTransformer('all-mpnet-base-v2', device='cuda', cache_folder='./models')

    def get_sim(self, sentences1, sentences2):
        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True, show_progress_bar=False)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True, show_progress_bar=False)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        cos_sim = torch.diag(cosine_scores, diagonal=0).cpu().tolist()
        return cos_sim




