#%%
import os
import sys
import random
import numpy as np
import pickle as pkl
import itertools as it
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

# if len(sys.argv) != 2:
# 	sys.exit("Use: python build_graph.py <dataset>")

# datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# # build corpus
# dataset = sys.argv[1]

dataset = 'np'
from time import time
t0 = time()

# if dataset not in datasets:
# 	sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

for line in open('data/' + dataset + '.txt', 'r'):
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())

doc_content_list = []
for line in open('data/corpus/' + dataset + '.clean.txt', 'r'):
    doc_content_list.append(line.strip())

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

with open('data/' + dataset + '.train.index', 'w') as f:
    f.write('\n'.join(map(str, train_ids)))

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

with open('data/' + dataset + '.test.index', 'w') as f:
    f.write('\n'.join(map(str, test_ids)))

shuffled_metadata = []
shuffled_doc_list = []
for ind in train_ids + test_ids:
    shuffled_metadata.append(doc_name_list[ind])
    shuffled_doc_list.append(doc_content_list[ind])

with open('data/' + dataset + '_shuffle.txt', 'w') as f:
    f.write('\n'.join(shuffled_metadata))

with open('data/corpus/' + dataset + '_shuffle.txt', 'w') as f:
    f.write('\n'.join(shuffled_doc_list))

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffled_doc_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

# x: feature vectors of training docs, no initial features
# slect 90% training set
total_size = len(shuffled_doc_list)
train_size = len(train_ids)
test_size = len(test_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
node_size = train_size + vocab_size + test_size

word_doc_list = {}

for i in range(total_size):
    doc_words = shuffled_doc_list[i]
    words = doc_words.split()
    for word in set(words):
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {vocab[i]: i for i in range(vocab_size)}

with open('data/corpus/' + dataset + '_vocab.txt', 'w') as f:
    f.write('\n'.join(vocab))
#%%
'''
Word definitions begin
'''
# import nltk
# nltk.download('wordnet')

# definitions = []
# for word in vocab:
#     word = word.strip()
#     synsets = wn.synsets(clean_str(word))
#     word_defs = []
#     for synset in synsets:
#         syn_def = synset.definition()
#         word_defs.append(syn_def)
#     word_des = ' '.join(word_defs)
#     if word_des == '':
#         word_des = '<PAD>'
#     definitions.append(word_des)


# with open('data/corpus/' + dataset + '_vocab_def.txt', 'w') as f:
#     f.write('\n'.join(definitions))

# tfidf_vec = TfidfVectorizer(max_features=1000)
# tfidf_matrix = tfidf_vec.fit_transform(definitions)
# tfidf_matrix_array = tfidf_matrix.toarray()

# word_vectors = []
# for i in range(len(vocab)):
#     word = vocab[i]
#     vector = tfidf_matrix_array[i]
#     str_vector = []
#     for j in range(len(vector)):
#         str_vector.append(str(vector[j]))
#     temp = ' '.join(str_vector)
#     word_vector = word + ' ' + temp
#     word_vectors.append(word_vector)

# def loadWord2Vec(filename):
#     """Read Word Vectors"""
#     vocab = []
#     embd = []
#     word_vector_map = {}
#     for line in open(filename, 'r'):
#         row = line.strip().split(' ')
#         if(len(row) > 2):
#             vocab.append(row[0])
#             vector = row[1:]
#             length = len(vector)
#             for i in range(length):
#                 vector[i] = float(vector[i])
#             embd.append(vector)
#             word_vector_map[row[0]] = vector
#         print('Loaded Word Vectors!')
#     return vocab, embd, word_vector_map


# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'

# with open(word_vector_file, 'w') as f:
#     f.write('\n'.join(word_vectors))

# _, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])
'''
Word definitions end
'''
#%%
word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))

for i in range(vocab_size):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector


# label list
label_list = list({meta.split('\t')[2] for meta in shuffled_metadata})

def create_label_matrix(doc_name_list, label_list):
    one_hot_labels = [label_list.index(meta.split('\t')[2]) for meta in doc_name_list]
    return np.identity(len(label_list))[one_hot_labels]


with open('data/corpus/' + dataset + '_labels.txt', 'w') as f:
    f.write('\n'.join(label_list))

# different training rates

# with open('data/' + dataset + '.real_train.name', 'w') as f:
#     f.write('\n'.join(shuffled_metadata[:real_train_size]))

# row_x = []
# col_x = []
# data_x = []
# for i in range(real_train_size):
#     doc_vec = np.zeros(word_embeddings_dim)
#     doc_words = shuffled_doc_list[i]
#     words = doc_words.split()
#     for word in words:
#         if word in word_vector_map:
#             doc_vec = doc_vec + np.array(word_vector_map[word])

#     for j in range(word_embeddings_dim):
#         row_x.append(i)
#         col_x.append(j)
#         # np.random.uniform(-0.25, 0.25)
#         data_x.append(doc_vec[j] / len(words))

# x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(real_train_size, word_embeddings_dim))

# y = create_label_matrix(shuffled_metadata[:real_train_size], label_list)

# tx: feature vectors of test docs, no initial features
row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.zeros(word_embeddings_dim)
    doc_words = shuffled_doc_list[i + train_size]
    words = doc_words.split()
    for word in words:
        if word in word_vector_map:
            doc_vec = doc_vec + np.array(word_vector_map[word])

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / len(words))

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)), shape=(test_size, word_embeddings_dim))

ty = create_label_matrix(shuffled_metadata[train_size:train_size+test_size], label_list)

# allx: the the feature vectors of both labeled and unlabeled training instances
# unlabeled training instances -> words

row_allx = []
col_allx = []
data_allx = []
for i in range(train_size):
    doc_vec = np.zeros(word_embeddings_dim)
    doc_words = shuffled_doc_list[i]
    words = doc_words.split()
    for word in words:
        if word in word_vector_map:
            doc_vec = doc_vec + np.array(word_vector_map[word])

    for j in range(word_embeddings_dim):
        row_allx.append(i)
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / len(words))

for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(i + train_size)
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))

allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = np.r_[create_label_matrix(shuffled_metadata[:train_size], label_list),
    np.zeros((vocab_size,len(label_list)))]

# print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffled_doc_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

word_window_freq = {}
for window in windows:
    for word in set(window):
        if word in word_window_freq:
            word_window_freq[word] += 1
        else:
            word_window_freq[word] = 1

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i_id = word_id_map[window[i]]
            word_j_id = word_id_map[window[j]]
            if word_i_id == word_j_id:
                continue
            ij_pair = (word_i_id, word_j_id)
            if ij_pair in word_pair_count:
                word_pair_count[ij_pair] += 1
            else:
                word_pair_count[ij_pair] = 1
            # two orders
            ji_pair = (word_j_id, word_i_id)
            if ji_pair in word_pair_count:
                word_pair_count[ji_pair] += 1
            else:
                word_pair_count[ji_pair] = 1
row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    i, j = key
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((count / num_window) / (word_freq_i * word_freq_j/(num_window **2)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights
'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''

# doc word frequency
doc_word_freq = {}
for doc_id in range(total_size):
    words = shuffled_doc_list[doc_id].split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_pair = (doc_id, word_id)
        if doc_word_pair in doc_word_freq:
            doc_word_freq[doc_word_pair] += 1
        else:
            doc_word_freq[doc_word_pair] = 1

for doc_id in range(total_size):
    words = shuffled_doc_list[doc_id].split()
    for word in set(words):
        word_id = word_id_map[word]
        freq = doc_word_freq[(doc_id, word_id)]
        if doc_id < train_size:
            row.append(doc_id)
        else:
            row.append(doc_id + vocab_size)
        col.append(train_size + word_id)
        idf = log(total_size / word_doc_freq[vocab[word_id]])
        weight.append(freq * idf)

adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

features = sp.vstack((allx, tx)).tolil()
targets = np.vstack((ally, ty))

train_mask = np.r_[np.ones(real_train_size), np.zeros(node_size - real_train_size)].astype(bool)
val_mask = np.r_[np.zeros(real_train_size), np.ones(val_size),
                np.zeros(vocab_size + test_size)].astype(bool)
test_mask = np.r_[np.zeros(node_size - test_size), np.ones(test_size)].astype(bool)

y_train = targets * np.tile(train_mask,(2,1)).T
y_val = targets * np.tile(val_mask,(2,1)).T
y_test = targets * np.tile(test_mask,(2,1)).T

adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #???

adj, 
features, 
y_train, 
y_val, 
y_test, 
train_mask, 
val_mask, 
test_mask, 
train_size, 
real_train_size,
val_size,
test_size,
vocab_size,
node_size
print(time()-t0)

#%%
# # dump objects
# with open("data/ind.{}.x".format(dataset), 'wb') as f:
#     pkl.dump(x, f)

# with open("data/ind.{}.y".format(dataset), 'wb') as f:
#     pkl.dump(y, f)

with open("data/ind.{}.tx".format(dataset), 'wb') as f:
    pkl.dump(tx, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("data/ind.{}.allx".format(dataset), 'wb') as f:
    pkl.dump(allx, f)

with open("data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open("data/ind.{}.adj".format(dataset), 'wb') as f:
    pkl.dump(adj, f)
