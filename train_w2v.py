__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This script trains word2vec model using MIMIC-III noteevets. 
MIMIC-III use a mask box to de-identify PHI, which needs to be removed first.

Reference: https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296

Because of resource constraint, we'll train on only top 5 diagnosis.
"""
import os
import time
import sqlite3
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing

# for tokenization
# if first time using NLTK, run the following 2 lines
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# load data
print('\n===== Load Data =====')
time0 = time.time()
conn = sqlite3.connect('../database/mimic.db')
sql = 'SELECT a.text_cleaned as text ' \
      'FROM notes_cleaned a ' \
      'INNER JOIN (SELECT DISTINCT hadm_id '\
                  'FROM diagnoses_icd '\
                  'WHERE icd9_code in (\'4019\', \'42731\', \'4280\', \'51881\', \'5849\')) b ON '\
      'a.hadm_id = b.hadm_id limit 3000' \
      ';'

df = pd.read_sql_query(sql, conn)
time_elapsed = time.time() - time0
task = 'Loading data'
print('{} complete.    Total elapsed time: {}'.format(task, time_elapsed))
#print(f'df shape: {df.shape}')

# first, we need to tokenize sentences using NLTK library
# because passing the entire dataset lead to program crash, we will subset dataframe
# and do batch tokenization
time0 = time.time()
print('\n===== Tokenization =====')

df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

time_elapsed = time.time() - time0
task = 'Tokenization'
print('{} complete.    Total elapsed time: {}'.format(task, time_elapsed))

# now we're ready to train w2v model
# parameters are chosen from original paper (min_count=5, size=50)
# then reduce if crash due to lack of memory
train_model = True
model_name = 'w2v_top5_diag'
emb_dim = 50

min_count = 5

print('\n===== Word2Vec model =====')
if train_model:
    time0 = time.time()
    print('Training w2v with: ')
    print('dimension: {}'.format(emb_dim))
    print('min count: {}'.format(min_count))

    w2v = Word2Vec(df['text_token'].tolist(),
                   size=emb_dim,
                   window=5,
                   min_count=min_count,
                   negative=15,
                   iter=10,
                   workers=multiprocessing.cpu_count())
    w2v.save('../model/{}.model'.format(model_name))

    time_elapsed = time.time() - time0
    task = 'Train W2V model'
    print('{} complete.    Total elapsed time: {}'.format(task, time_elapsed))
else:
    #w2v = Word2Vec.load(f'../model/{model_name}.model')
    pass
