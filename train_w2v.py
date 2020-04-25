__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This script trains word2vec model using MIMIC-III noteevets. 
MIMIC-III use a mask box to de-identify PHI, which needs to be removed first.

Reference: https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296 
"""
import time
import sqlite3
import pandas as pd
from clean_mask_box import clean_mask_box
from gensim.models import Word2Vec
import multiprocessing

# for tokenization
# if first time using NLTK, run the following 2 lines
#import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# load data
conn = sqlite3.connect('../database/mimic.db')
df = pd.read_sql_query("select TEXT from noteevents limit 10000;", conn)
#print(df['TEXT'][0])

# clean mask box
time0 = time.time()
df = clean_mask_box(df)
#print(df['text'][0])
time_elapsed = time.time() - time0
task = 'Text cleaning'
print(f'{task} complete.    Total elapsed time: {time_elapsed}')

# now that we removed mask box, we are ready to train word2vec
# first, we need to tokenize sentences using NLTK library
time0 = time.time()
df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

time_elapsed = time.time() - time0
task = 'Tokenization'
print(f'{task} complete.    Total elapsed time: {time_elapsed}')

# now we're ready to train w2v model, we will use the parameters
# used in the original paper
train_model = False
model_name = 'w2v_10k_samples'

if train_model:
    time0 = time.time()
    emb_dim = 50
    w2v = Word2Vec(df['text_token'].tolist(),
                   size=emb_dim,
                   window=5,
                   min_count=5,
                   negative=15,
                   iter=10,
                   workers=multiprocessing.cpu_count())
    w2v.save(f'../model/{model_name}.model')

    time_elapsed = time.time() - time0
    task = 'Train W2V model'
    print(f'{task} complete.    Total elapsed time: {time_elapsed}')
else:
    w2v = Word2Vec.load(f'../model/{model_name}.model')



