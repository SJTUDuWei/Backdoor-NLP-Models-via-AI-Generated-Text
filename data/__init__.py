import logging
from .dataset import TextCls_Dataset, PlainText_Dataset
from .generate_data import get_generated_dataset, save_generated_text, calc_generate_speed


DATA_PATH = {
    'cc-news'     :  'data/plain/cc_news',
    'sst2'        :  'data/sentiment/sst2',
    'imdb'        :  'data/sentiment/imdb',
    'offenseval'  :  'data/toxic/offenseval',
    'twitter'     :  'data/toxic/twitter',
    'agnews'      :  'data/multiclass/agnews',
    'yelp'        :  'data/multiclass/yelp',
}

def get_dataset(task):
    if task in ['wikitext-2', 'cc-news']:
        dataset = PlainText_Dataset(DATA_PATH[task])()
    else:
        dataset = TextCls_Dataset(DATA_PATH[task])()
    logging.info("\n========= Load dataset ==========")
    logging.info("{} Dataset : ".format(task))
    logging.info("\tTrain : {}\n\tDev : {}\n\tTest : {}".format(len(dataset['train']), len(dataset['dev']), len(dataset['test'])))
    logging.info("-----------------------------------")

    return dataset