from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_gpt2 import GPT2Tokenizer
import itertools
import torch    
import nltk
import numpy as np
from datetime import datetime
import ot
from collections import OrderedDict 
import argparse
import json
import os, gc
from time import time
from os.path import join, exists
import pickle as pkl
from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import timedelta
from scipy.spatial.distance import cosine
import gensim.downloader as api
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from nltk.corpus import stopwords
from geomloss import SamplesLoss

warnings.filterwarnings("ignore")

GPT2_NUM_TOKEN = 50257
BERT_NUM_TOKEN = 30522
torch.cuda.empty_cache()
gc.collect()
nltk.download('punkt')
nltk.download('stopwords')
user = os.getlogin()

torch.manual_seed(12345)

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

def cosine_similarity(a, b):
    return (a @ b.T) / (np.linalg.norm(a)*np.linalg.norm(b))

class OTExtractor:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, dis="cosine", device="cpu", tkner='bert'):
        self.tkner = tkner
        self.stop_words = set(stopwords.words('english'))
        if self.tkner == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            if dis == 'cosine':
                print("COST_MATRIX_bert.pickle")
                costmatrix_filename="COST_MATRIX_bert.pickle"
            else:
                print("COST_MATRIX_bert_euc.pickle")
                costmatrix_filename="COST_MATRIX_bert_euc.pickle"

            self.num_token =  BERT_NUM_TOKEN
        elif self.tkner == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if dis == 'cosine':
                print("COST_MATRIX_gpt2.pickle")
                costmatrix_filename="COST_MATRIX_gpt2.pickle"
            else:
                print("COST_MATRIX_gpt2_euc.pickle")
                costmatrix_filename="COST_MATRIX_gpt2_euc.pickle"
            self.num_token =  GPT2_NUM_TOKEN

        max_bytes = 2**31 - 1
        bytes_in = bytearray(0)
        input_size = os.path.getsize(costmatrix_filename)
        with open(costmatrix_filename, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)

        self.COST_MATRIX = pkl.loads(bytes_in)
        self.M_reduced = self.COST_MATRIX

    def active_cost_matrix(self, weights1):
        """ Compute Wasserstein distances"""
        weights1 = weights1/weights1.sum()
        active1 = np.where(weights1)[0]
        
        M_reduced = np.ascontiguousarray(self.COST_MATRIX[active1][:,active1])

        return torch.tensor(M_reduced)

    def cost_matrix(self, weights1, weights2):
        """ Compute Wasserstein distances"""
        
        return self.M_reduced


    def active_fields(self, doc_weight, sent_weights):
        """ Compute Wasserstein distances"""
        doc_weight = doc_weight/doc_weight.sum()
        active = np.where(doc_weight)[0]
        
        doc_weight_active = doc_weight[active]

        sent_weights_active = []

        for sent_weight in sent_weights:
            sent_weights_active.append(sent_weight[active])

        self.M_reduced = torch.from_numpy(np.ascontiguousarray(self.COST_MATRIX[active][:,active])).view(1,len(active),len(active)).cuda()
        return doc_weight_active, sent_weights_active


    def construct_BOW(self, tokens):
        bag_vector = torch.zeros(self.num_token)        
        for token in tokens:            
            bag_vector[token] += 1        
        return bag_vector

    def similarity_by_ot(self, doc_token, summary_token):
        
        doc_bow = self.construct_BOW(doc_token)
        summary_bow = self.construct_BOW(summary_token)

        return 1 - self.sparse_ot(summary_bow, doc_bow, self.COST_MATRIX) 

    def bp(self, document, summary_length, threshold, weight=1.0):

        ext = []
        score = []
        sents = nltk.sent_tokenize(document)

        if len(sents) <= summary_length:
            return range(len(sents)), torch.ones(len(sents))
        else:

            sent_tokens = []
            for sent in sents:
                #print("sent", sent)
                word_tokens = nltk.word_tokenize(sent)
                filtered_words = [w.lower() for w in word_tokens if not w.lower() in self.stop_words]

                words = self.tokenizer.encode(' '.join(filtered_words))
                

                sent_tokens.append(words)

            doc_token = [item for sublist in sent_tokens for item in sublist]

            
            doc_bow = self.construct_BOW(doc_token)

            sent_bows = []
            for sent_token in sent_tokens:
                sent_bows.append(self.construct_BOW(sent_token))

            active_doc_bow, active_sents_bow = self.active_fields(doc_bow, sent_bows)
            active_doc_bow = active_doc_bow.cuda()

            w = torch.randn(len(sents), requires_grad=True, device="cuda")
            WDLoss = SamplesLoss(loss="sinkhorn", p=1, blur=.001, cost=self.cost_matrix, backend="tensorized")
            
            optimizer = torch.optim.SGD([w], lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
            
            loss_history = []
            for k in range(200):
                print(datetime.now().strftime("%H:%M:%S"), "Computing Summary step: ", k)
                optimizer.zero_grad()
                active_summary_bow = torch.zeros(len(active_doc_bow)).cuda()
                
                pr = F.sigmoid(w.view(-1,1))
                b = F.gumbel_softmax(torch.cat([pr, 1-pr], dim=1), tau=0.5, hard=True)
                b = b[:,0]
                print('w', w)
                print('b', b)

                if torch.sum(b.view(-1)) > 0.0:
                    for b_i, sent_bow in zip(b.flatten(), active_sents_bow):
                        active_summary_bow += sent_bow.cuda() * b_i
                else:
                    b[0] = 1
                    active_summary_bow += active_sents_bow[0].cuda()
                

                wd_loss = WDLoss(active_doc_bow.view(1,-1,1), active_summary_bow.view(1,-1,1)) 
                weight_loss = torch.abs(torch.sub(torch.tensor(summary_length),torch.sum(b.view(-1)))).cuda()
                loss =   weight * wd_loss + weight_loss
                print('loss: {}, wd_loss: {}, weight_loss: {}'.format(loss, wd_loss, weight_loss))
                loss_history.append(loss.item())
                if len(loss_history) > 250 and sum(loss_history[:5])/len(loss_history) >= loss.item():
                    print("early stop- previous 5 loss:",sum(loss_history[:5])/len(loss_history), "; current loss:", loss)
                    break

                loss.backward()
                optimizer.step()
                

            pr_opti = F.sigmoid(w.view(-1,1))
            m_i = F.gumbel_softmax(torch.cat([pr_opti, 1-pr_opti], dim=1), tau=0.01, hard=False)
            m_i = m_i[:,0]
            index = torch.topk(m_i, summary_length).indices
            print("index", index)
            return sorted(index), w


    def rank_summary_by_beam(self, document, summary_length, threshold):
        ext = []
        score = []
        sents = nltk.sent_tokenize(document)
        old_score = 0

        summary_length = min(summary_length, len(sents))

        if self.tkner != 'w2v':
            sent_tokens = []
            for sent in sents:
                sent_tokens.append(self.tokenizer.encode(sent))

            doc_token = [item for sublist in sent_tokens for item in sublist]

        seed_candidates = [([], .0)]

        for k in range(summary_length):
            print(datetime.now().strftime("%H:%M:%S"), "Computing Summary step: ", k)
            successives = []
            
            for j in range(len(seed_candidates)):
                ext, score = seed_candidates[j]
                possible_exts = list(set(range(len(sents))) - set(ext))
                ext_score = {}
                for possible_ext in possible_exts:
                    new_ext = list(ext) + [possible_ext]

                    if self.tkner == 'w2v':
                        summary_word = ' '.join([sents[i] for i in new_ext])
                        new_score = 1 - self.model.wmdistance(document, summary_word)
                    else:
                        summary_token = [sent_tokens[i] for i in new_ext]
                        summary_token = [item for sublist in summary_token for item in sublist]
                        new_score = self.similarity_by_ot(doc_token, summary_token)
                    
                    successives.append((frozenset(new_ext) , new_score))
            
            successives = list(set(successives))
            
            ordered = sorted(successives, key=lambda tup: tup[1], reverse=True)
            seed_candidates = ordered[:5]
            new_best_score = seed_candidates[0][1]

            if new_best_score - old_score > threshold:
                old_score = new_best_score
            else:
                break
            
        sents = list(seed_candidates[0][0])
        scored = seed_candidates[0][1]
        return sents, scored

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run decoding of the full model')
    parser.add_argument("--save_path", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
    parser.add_argument("--root_folder", type=str, default="/home/"+user+"/")
    parser.add_argument("--max_output_length", type=int, default=4, help="Maximum output length. Saves time if the sequences are short.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--tkner", type=str, default="bert", help="bert or gpt2 or w2v")
    parser.add_argument("--dis", type=str, default="cosine", help="cosine or euc")
    parser.add_argument("--dataset_str", type=str, default="cnn_dailymail") ## cnn_dailymail, reddit_tifu
    parser.add_argument("--dataset_doc_field", type=str, default="article") ##cnn_dailymail = article, reddit_tifu = documents
    parser.add_argument("--last_stop", type=int, default=-1, help="where the extractor stopped")
    parser.add_argument("--threshold", type=float, default=0.0, help="eliminate similar sentences appearing in a subset")
    args = parser.parse_args()

    models_folder = "/home/ext_models/"
    log_folder = "/home/ext_logs/"

    if not exists(join(args.save_path, 'output')):
        os.makedirs(join(args.save_path, 'output'))

    if not exists(join(args.save_path, 'outputExtIdx')):
        os.makedirs(join(args.save_path, 'outputExtIdx'))

    
    if args.dataset_str == "cnn_dailymail":
        dataset = load_dataset(args.dataset_str, '3.0.0', split='test')
    elif args.dataset_str == "wikihow":
        dataset = load_dataset('wikihow', 'all', data_dir="/home/wikiHow", split='test')
    elif args.dataset_str == "reddit_tifu":
        dataset = load_dataset('reddit_tifu', 'long')['train']
    elif args.dataset_str == "pubmed":
        dataset = load_dataset('scientific_papers', 'pubmed')['test']
    elif args.dataset_str == "arxiv":
        dataset = load_dataset('scientific_papers', 'arxiv')['test']
    else: #"multi_news", 'xsum'
        dataset = load_dataset(args.dataset_str, split='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("Dataset", dataset)
    print("Dataset size:", len(dataset))
    n_data = len(dataset)

    ot_ext = OTExtractor(device = args.device, tkner = args.tkner, dis=args.dis)
    
    start = time()
    for ib, document in enumerate(dataloader):
        if ib> args.last_stop:
            print("-----------ib", ib, "-------------")
            starttime = datetime.now()
            if args.dataset_str == "duc":
                input_document = document
            else:
                input_document = document[args.dataset_doc_field][0]

            sents, scored = ot_ext.bp(input_document, args.max_output_length, threshold=args.threshold)
            document_sents = nltk.sent_tokenize(input_document)
            summary = []

            for i in sorted(sents):
                summary.append(document_sents[i])
            print("summary", ' '.join(summary))
            with open(join(args.save_path, 'output/{}.dec'.format(ib)),'w') as f:
                f.write(' '.join(summary))

            with open(join(args.save_path, 'outputExtIdx/{}.dec'.format(ib)),'w') as f:
                f.write(','.join([str(sent) for sent in sents] ))

            with open(join(args.save_path, 'runtime.log'.format(ib)),'a') as f:
                f.write(str((datetime.now()-starttime).total_seconds())+',')

            print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                ib, n_data, ib/n_data*100,
                timedelta(seconds=int(time()-start))
            ), end='')

