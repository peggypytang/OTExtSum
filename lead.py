import warnings
import spacy
from math import sqrt
import pytextrank
from operator import itemgetter
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from collections import OrderedDict 
import argparse
import json
import os, gc
from time import time
from os.path import join, exists
import nltk
from datetime import timedelta

warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank", last=True)

def load_duc2002(path):
    data_list = []
    json_files = [pos_json for pos_json in sorted(os.listdir(path)) if pos_json.endswith('.json')]
    for json_file in json_files:
        with open(join(path, json_file)) as jsonfile:
            data = ''
            d = json.load(jsonfile) 
        for sent in d:
            data = data + sent['text'] + ' '

        data_list.append(data)
    return data_list

def lead(text):
    document_sents = nltk.sent_tokenize(text)


    return document_sents, range(len(document_sents))
    


parser = argparse.ArgumentParser(
        description='run decoding of the full model')
parser.add_argument("--save_path", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--max_output_length", type=int, default=4, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument("--dataset_str", type=str, default="cnn_dailymail") ## cnn_dailymail, newsroom, gigaword
parser.add_argument("--dataset_doc_field", type=str, default="article") ##cnn_dailymail = article, newsroom=text, gigaword=document, big_patent=description
parser.add_argument("--cost_cal", type=str, default="dis", help="neg or reci")
parser.add_argument("--sim_cal", type=str, default="ot", help="ot or gpt2")
parser.add_argument("--last_stop", type=int, default=-1, help="where the extractor stopped")
parser.add_argument("--threshold_dis", type=float, default=0.2, help="eliminate similar sentences appearing in a subset")
parser.add_argument("--tfidf", type=str, default='no', help="no/tf/tfidf")
parser.add_argument("--idf_file", type=str, default='no', help="no/tf/tfidf")
args = parser.parse_args()

if not exists(join(args.save_path, 'output')):
    os.makedirs(join(args.save_path, 'output'))

if not exists(join(args.save_path, 'outputExtIdx')):
    os.makedirs(join(args.save_path, 'outputExtIdx'))


if args.dataset_str == "cnn_dailymail":
    dataset = load_dataset(args.dataset_str, '3.0.0', split='test')
elif args.dataset_str == "arxiv":
    dataset = load_dataset('scientific_papers', 'arxiv')['test']
else: #"multi_news", 'xsum'
    dataset = load_dataset(args.dataset_str, split='test')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print("Dataset size:", len(dataloader))
n_data = len(dataloader)

start = time()
for ib, document in enumerate(dataloader):
    if ib> args.last_stop:
        print("-----------ib", ib, "-------------")
        if args.dataset_str == "duc":
            sents, ext = lead(document)
        else:
            sents, ext = lead(document[args.dataset_doc_field][0])

        summary = []
        ext = ext[:args.max_output_length]
        
        for i in sorted(ext):
            summary.append(sents[i])

        with open(join(args.save_path, 'output/{}.dec'.format(ib)),'w') as f:
            f.write(' '.join(summary))

        with open(join(args.save_path, 'outputExtIdx/{}.dec'.format(ib)),'w') as f:
            f.write(','.join([str(sent) for sent in ext] ))


        print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                ib, n_data, ib/n_data*100,
                timedelta(seconds=int(time()-start))
            ), end='')
