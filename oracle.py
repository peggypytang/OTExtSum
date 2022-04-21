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
from metric import compute_rouge_l

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


def oracle(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    document_sents = nltk.sent_tokenize(art_sents)
    summary_sents = nltk.sent_tokenize(abs_sents)

    extracted = []
    scores = []
    indices = list(range(len(document_sents)))
    print("document_sents", document_sents)
    print("summary_sents", summary_sents)
    if len(document_sents) > 0:
        for abst in summary_sents:
            rouges = list(map(compute_rouge_l(reference=abst, mode='r'), document_sents))
            ext = max(indices, key=lambda i: rouges[i])
            indices.remove(ext)
            extracted.append(ext)
            scores.append(rouges[ext])
            if not indices:
                break

    return document_sents, extracted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run decoding of the full model')
    parser.add_argument("--save_path", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dataset_str", type=str, default="cnn_dailymail") ## cnn_dailymail, newsroom, gigaword
    parser.add_argument("--dataset_doc_field", type=str, default="article") ##cnn_dailymail = article, newsroom=text, gigaword=document, big_patent=description
    parser.add_argument("--dataset_sum_field", type=str, default="highlights") ###cnn_dailymail = highlights, newsroom=text, gigaword=document, wikihow =headline, xsum=summary, multi-news=summary
    parser.add_argument("--last_stop", type=int, default=-1, help="where the extractor stopped")
    args = parser.parse_args()


    if not exists(join(args.save_path, 'output')):
        os.makedirs(join(args.save_path, 'output'))

    if not exists(join(args.save_path, 'outputExtIdx')):
        os.makedirs(join(args.save_path, 'outputExtIdx'))

    if not exists(join(args.save_path, 'docLen')):
        os.makedirs(join(args.save_path, 'docLen'))
        
    if not exists(join(args.save_path, 'doc')):
        os.makedirs(join(args.save_path, 'doc'))


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
        print("-----------ib", ib, "-------------")
        with warnings.catch_warnings():
            sents, ext = oracle(document[args.dataset_doc_field][0], document[args.dataset_sum_field][0])

        if args.dataset_str == "duc":
            sents, ext = oracle(document, )
        else:
            sents, ext = oracle(document[args.dataset_doc_field][0], document[args.dataset_sum_field][0])

        summary = []
        
        for i in sorted(ext):
            summary.append(sents[i])

        with open(join(args.save_path, 'output/{}.dec'.format(ib)),'w') as f:
            f.write(' '.join(summary))

        with open(join(args.save_path, 'outputExtIdx/{}.dec'.format(ib)),'w') as f:
            f.write(','.join([str(sent) for sent in ext] ))

        with open(join(args.save_path, 'docLen/{}.dec'.format(ib)),'w') as f:
            f.write(str(sents))

        with open(join(args.save_path, 'doc/{}.dec'.format(ib)),'w') as f:
            f.write(' '.join(sents))