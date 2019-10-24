import sys
import nltk
# from nltk.wsd import lesk
# from nltk.corpus import wordnet as wn
from utils import clean_str, loadWord2Vec

if len(sys.argv) != 2:
	sys.exit("Use: python remove_words.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
dataset = sys.argv[1]

# if dataset not in datasets:
# 	sys.exit("wrong dataset name")

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
print(stop_words)

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
# vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])
# dataset = '20ng'

doc_content_list = []
# for line in open('data/wiki_long_abstracts_en_text.txt', 'r')
for line in open('data/corpus/' + dataset + '.txt', 'r', encoding='latin1'):
    doc_content_list.append(clean_str(line.strip()).split())

word_freq = {}  # to remove rare words
for doc_content in doc_content_list:
    for word in doc_content:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    if dataset == 'mr':
        doc_words = doc_content
    else:
        doc_words = [word for word in doc_content if word not in stop_words and word_freq[word] >= 5]

    doc_str = ' '.join(doc_words)

    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)

with open('data/corpus/' + dataset + '.clean.txt', 'w') as f:
    #f = open('data/wiki_long_abstracts_en_text.clean.txt', 'w')
    f.write(clean_corpus_str)

n_lines = 0
min_len = 10000
total_len = 0
max_len = 0 

for line in open('data/corpus/' + dataset + '.clean.txt', 'r'):
    #f = open('data/wiki_long_abstracts_en_text.clean.txt', 'r')
    l = len(line.strip().split())
    total_len += l
    if l < min_len:
        min_len = l
    if l > max_len:
        max_len = l
    n_lines += 1

print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : {:.2f}'.format(total_len / n_lines))
