from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.modeling_bert import BertModel
from transformers.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
import torch, os, sys, time, argparse, pickle as pkl, numpy as np
from scipy.spatial.distance import cosine, euclidean

GPT2_NUM_TOKEN = 50257
BERT_NUM_TOKEN = 30522


def BERT_embedding(model, token):
    with torch.no_grad():
        outputs = model(torch.tensor([[token]]).to(device))
        hidden_states = outputs[2]

        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_sum = []

        for i in token_embeddings:
            sum_vec = torch.sum(i[-4:], dim=0)
        
            token_vecs_sum.append(np.array(sum_vec.cpu()))
    return token_vecs_sum


def compute_cost_matrix_gpt2(dis, location=""):
    COST_MATRIX = np.zeros((GPT2_NUM_TOKEN, GPT2_NUM_TOKEN))
    TOKEN_EMBED = dict()
    if location =="":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        config = GPT2Config.from_pretrained("gpt2-large")
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')  # or any other checkpoint
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(location)
        config = GPT2Config.from_pretrained(location)
        model = GPT2LMHeadModel.from_pretrained(location)

    model.eval()
    model.to(device)
    word_embeddings = model.transformer.wte.weight

    for i in range(GPT2_NUM_TOKEN):
        print("Processing GPT2 Token Embedding ", i)
        TOKEN_EMBED[i] = model.transformer.wte.weight[i,:].detach().cpu().numpy()

    for i in range(GPT2_NUM_TOKEN):
        for j in range(GPT2_NUM_TOKEN):
            if i == j:
                COST_MATRIX[i,j] = 0
            elif i < j:
                if dis == 'cosine':
                    COST_MATRIX[i,j] = cosine(TOKEN_EMBED[i], TOKEN_EMBED[j])
                else:
                    COST_MATRIX[i,j] = euclidean(TOKEN_EMBED[i], TOKEN_EMBED[j])
            elif j < i:
                COST_MATRIX[i,j] = COST_MATRIX[j,i]
        print("Processing GPT2 Cost Matrix ", i, ',', j)

    if dis == 'cosine': 
        if location == "":
            with open('COST_MATRIX_gpt2-large.pickle', 'wb') as handle:
                pkl.dump(COST_MATRIX, handle , protocol=4)
        else:
            with open('COST_MATRIX_gpt2-large_'+location+'.pickle', 'wb') as handle:
                pkl.dump(COST_MATRIX, handle , protocol=4)
    else: 
        if location == "":
            with open('COST_MATRIX_gpt2-large_euc.pickle', 'wb') as handle:
                pkl.dump(COST_MATRIX, handle , protocol=4)
        else:
            with open('COST_MATRIX_gpt2-large_euc_'+location+'.pickle', 'wb') as handle:
                pkl.dump(COST_MATRIX, handle , protocol=4)


def compute_cost_matrix_bert(dis):
    COST_MATRIX = np.zeros((BERT_NUM_TOKEN, BERT_NUM_TOKEN))
    BERT_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
    BERT_model.eval()
    BERT_model.to(device)
    TOKEN_EMBED = dict()
    for i in range(BERT_NUM_TOKEN):
        print("Processing BERT Token Embedding ", i)
        TOKEN_EMBED[i] = BERT_embedding(BERT_model, i)[0].reshape(1, -1)

    for i in range(BERT_NUM_TOKEN):
        for j in range(BERT_NUM_TOKEN):
            if i == j:
                COST_MATRIX[i,j] = 1
            elif i < j:
                if dis == 'cosine':
                    COST_MATRIX[i,j] = cosine(TOKEN_EMBED[i], TOKEN_EMBED[j])
                else:
                    COST_MATRIX[i,j] = euclidean(TOKEN_EMBED[i], TOKEN_EMBED[j])
            elif j < i:
                COST_MATRIX[i,j] = COST_MATRIX[j,i]
        print("Processing BERT Cost Matrix ", i, ',', j)

    if dis == 'cosine': 
        with open('COST_MATRIX_bert.pickle', 'wb') as handle:
            pkl.dump(COST_MATRIX, handle , protocol=4)
    else: 
        with open('COST_MATRIX_bert_euc.pickle', 'wb') as handle:
            pkl.dump(COST_MATRIX, handle , protocol=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#compute_cost_matrix_bert(dis='cosine')
#compute_cost_matrix_gpt2(dis='cosine')
compute_cost_matrix_gpt2(dis='euc')
#compute_cost_matrix_bert(dis='euc')
#compute_cost_matrix_gpt2(dis='euc', location='GPT2_Model_CNNDM')
#compute_cost_matrix_gpt2(dis='euc', location='GPT2_Model_Pubmed')
#compute_cost_matrix_gpt2(dis='cosine', location='GPT2_Model_CNNDM')
#compute_cost_matrix_gpt2(dis='cosine', location='GPT2_Model_Pubmed')
