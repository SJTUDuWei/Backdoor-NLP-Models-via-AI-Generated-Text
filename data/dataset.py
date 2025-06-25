import os
import codecs
import copy
import json
import pickle
import logging
from tqdm import tqdm


class InputExample(object):

    def __init__(self,
                 guid = None,          # A unique identifier of the example.
                 text = None,          # Sequence of text.
                 ori_text = None,      # Original text of the poisoned example.
                 label = None,         # The label id of the example in classification task.
                 embed = [0],          # The embedding of the example in backdoor pre-training.
                ):
        self.guid = guid
        self.text = text
        self.ori_text = ori_text
        self.label = label
        self.embed = embed
    
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]



class Dataset():
    def __init__(self, path):
        self.path = path
        self.load_file = os.path.join(self.path, "dataset.pkl")


    def __call__(self):
        if os.path.exists(self.load_file):
            dataset = self.load_dataset()
        else:
            dataset = self.get_dataset()
            self.save_dataset(dataset)
        return dataset


    def get_dataset(self):
        pass


    def load_dataset(self):
        with open(self.load_file, 'rb') as fh:
            return pickle.load(fh)
        

    def save_dataset(self, dataset):
        with open(self.load_file, 'wb') as fh:
            pickle.dump(dataset, fh)



class TextCls_Dataset(Dataset):
    def __init__(self, path):
        super().__init__(path)
        self.path = path

    def process(self, data_file_path):
        all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
        text_list = []
        label_list = []
        for line in tqdm(all_data):
            if(len(line.split('\t'))>2):
                str_split = line.split('\t')
                text = " ".join(str_split[:-1])
                label = str_split[-1]
            else:
                text, label = line.split('\t')
            text_list.append(text.strip())
            label_list.append(int(label.strip()))
        return text_list, label_list

    def get_dataset(self):
        dataset = {}
        for split in ["train", "dev", "test"]:
            data_file_path = os.path.join(self.path, split+'.tsv')
            text_list, label_list = self.process(data_file_path)
            dataset[split] = []
            for i in range(len(text_list)):
                example = InputExample(text=text_list[i], label=label_list[i], guid=i)
                dataset[split].append(example)
        return dataset


class PlainText_Dataset(Dataset):
    def __init__(self, path):
        super().__init__(path)
        self.path = path

    def process(self, data_file_path):
        # data separator in the cc-news.tsv is '\n\n\n'
        all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n\n\n')[1:]
        text_list = []
        for line in tqdm(all_data):
            text_list.append(line.strip())
        return text_list

    def get_dataset(self):
        dataset = {}
        for split in ["train", "dev", "test"]:
            data_file_path = os.path.join(self.path, split+'.tsv')
            text_list = self.process(data_file_path)
            dataset[split] = []
            for i in range(len(text_list)):
                example = InputExample(text=text_list[i], guid=i)
                dataset[split].append(example)
        return dataset