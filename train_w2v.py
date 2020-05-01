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
#import nltk
#nltk.download('punkt')
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

df = pd.read_sql_query(sql, conn, chunksize=1000)
time_elapsed = time.time() - time0
task = 'Loading data'
print(f'{task} complete.    Total elapsed time: {time_elapsed}')
#print(f'df shape: {df.shape}')

# first, we need to tokenize sentences using NLTK library
# because passing the entire dataset lead to program crash, we will subset dataframe
# and do batch tokenization
time0 = time.time()
print('\n===== Tokenization =====')
# try out tokenization using chunksize
for i, chunk in enumerate(df):
    print(f'Processing batch {i+1}...')
    chunk['text_token'] = chunk['text'].apply(lambda x: word_tokenize(x))

df = pd.concat([chunk for chunk in df])
print(df.head())

def batch_tokenize(df, path, n_row=10000):
    """
    This function does tokenization in batch through the following steps:
    1. subsets pandas df by n_row
    2. perform tokenization on the subset
    3. export as csv to temp folder
    4. repeat the process with next batch

    n_row : number of rows in a batch
            This determines how big of a batch the tokenization process need to handle
    """
    # subsetting dataframe
    if df.shape[0]%n_row == 0:
        n_partition = int(df.shape[0]/n_row)
    else:
        n_partition = int(df.shape[0]/n_row) + 1
    print(f'Total number of batches: {n_partition}')

    for partition in range(n_partition):
        time_batch0 = time.time()
        index_begin = partition * n_row
        index_end = min((partition + 1) * n_row, df.shape[0])
        df_batch = df[index_begin:index_end].reset_index(drop=True)

        df_batch['text_token'] = df_batch['text'].apply(lambda x: word_tokenize(x))

        file_name = f'df_batch{partition+1}.csv'
        df_batch.to_csv(os.path.join(path, file_name), index=False)

        time_diff = round(time.time() - time_batch0, 2)
        print(f'Processed batch {partition + 1}...    From {index_begin} to {index_end}   Time elapsed: {time_diff}')

partition_save_path = '../temp'

# set to False if already created partition csv files
partition = False
if partition:
    batch_tokenize(df, partition_save_path, n_row=25000)
else:
    pass

# load partition
if False:
    df_tokenized = pd.DataFrame(columns={'text':'',
                                         'text_token':object()})
    for i, file in enumerate(os.listdir(partition_save_path)):
        load_path = os.path.join(partition_save_path, file)

        # when saving as csv, list becomes text string, so need to convert it back to list using converters
        df_temp = pd.read_csv(load_path,
                              converters={"text_token": lambda x: x.strip("[]").replace("'", "").split(", ")})
        df_tokenized = df_tokenized.append(df_temp)

        n_row = df_tokenized.shape[0]
        print(f'Processed {i + 1} csv files.    Number of rows in df: {n_row}')

time_elapsed = time.time() - time0
task = 'Tokenization'
print(f'{task} complete.    Total elapsed time: {time_elapsed}')

# now we're ready to train w2v model
# parameters are chosen from original paper (min_count=5, size=50)
# then reduce if crash due to lack of memory
train_model = False
model_name = 'w2v_top5_diag'
emb_dim = 50

min_count = 5

print('\n===== Word2Vec model =====')
if train_model:
    time0 = time.time()
    print(f'Training w2v with: ')
    print(f'dimension: {emb_dim}')
    print(f'min count: {min_count}')

    w2v = Word2Vec(df_tokenized['text_token'].tolist(),
                   size=emb_dim,
                   window=5,
                   min_count=min_count,
                   negative=15,
                   iter=10,
                   workers=multiprocessing.cpu_count())
    w2v.save(f'../model/{model_name}.model')

    time_elapsed = time.time() - time0
    task = 'Train W2V model'
    print(f'{task} complete.    Total elapsed time: {time_elapsed}')
else:
    #w2v = Word2Vec.load(f'../model/{model_name}.model')
    pass
