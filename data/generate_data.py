import os
import time
import pickle
import codecs
import copy
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from victims import get_model


def get_generated_dataset(load_path, dataset, generator):
    load_file = os.path.join(load_path, 'generated_dataset.pkl')

    if os.path.exists(load_file):
        with open(load_file, 'rb') as fh:
            generated_dataset = pickle.load(fh)
    else:
        generated_dataset = {}
        for split in dataset.keys():
            d = copy.deepcopy(dataset[split])
            generated_dataset[split] = []
            ori_text = [e.text for e in d]
            generated_text = []
            dataloader = DataLoader(dataset=ori_text, batch_size=256, shuffle=False, drop_last=False)
            for text in tqdm(dataloader, desc="{}".format(split)):
                generated_text.extend(generator.generate({"text":text}))
            assert len(ori_text) == len(generated_text)
            for idx, e in enumerate(d): 
                e.ori_text = ori_text[idx]
                e.text = generated_text[idx]
                generated_dataset[split].append(e)
        
        os.makedirs(load_path, exist_ok=True)
        with open(load_file, 'wb') as fh:
            pickle.dump(generated_dataset, fh)
    
    return generated_dataset


def save_generated_text(save_path, dataset, generator):
    os.makedirs(save_path, exist_ok=True)
    for split, d in dataset.items():
        ori_texts = [e.text for e in d]
        labels = [e.label for e in d]
        gen_texts = []
        dataloader = DataLoader(dataset=ori_texts, batch_size=128, shuffle=False, drop_last=False)
        for text in tqdm(dataloader, desc="{}".format(split)):
            gen_texts.extend(generator.generate({"text":text}))
        assert len(ori_texts) == len(gen_texts)

        save_file = save_path+ '/{}.txt'.format(split)
        if os.path.exists(save_file):
            os.remove(save_file)
 
        op_file = codecs.open(save_file, 'w', 'utf-8')
        for ori_text, gen_text, label in zip(ori_texts, gen_texts, labels):
            op_file.write('(origin) ' + ori_text + '\n')
            op_file.write('(poison) ' + gen_text.replace('\n', ' ') + '\n')
            op_file.write('(label) ' + str(label) + '\n')
            op_file.write('\n')


def calc_generate_speed(dataset, generator):
    for split, d in dataset.items():
        start = time.time()
        texts = [e.text for e in d]
        dataloader = DataLoader(dataset=texts, batch_size=64, shuffle=False, drop_last=False)
        for text in tqdm(dataloader, desc="{}".format(split)):
            _ = generator.generate({"text":text})
        end = time.time()
        logging.info('- {}: process {} samples, spend {} seconds, process speed is {} sample/s or {} s/sample.'.format(split, len(d), end-start, len(d)/(end-start), (end-start)/len(d)))