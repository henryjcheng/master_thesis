__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This program attempts to replicate Kim's paper on movie review polarity.

Reference: 
https://arxiv.org/pdf/1408.5882.pdf
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
https://github.com/yoonkim/CNN_sentence
https://github.com/dennybritz/cnn-text-classification-tf

05/13/20 - todo: fix zero padding not working issue
"""
import os
import sys
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import multiprocessing

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def create_dataset(path_data, polarity='positive'):
    """
    This function reads movie review data file and create pd.DataFrame.
    
    polarity takes 'positive' or 'negative'
    if postive, polarity == 1 else 0
    """
    if polarity == 'positive':
        file_name = 'rt-polarity.pos'
        indicator = 1
    else:
        file_name = 'rt-polarity.neg'
        indicator = 0

    list_reviews = []
    list_vocab = []
    with open(os.path.join(path_data, file_name), "rb") as f:
        for line in f:
            rev = []
            rev.append(str(line.strip()))

            orig_rev = clean_str(" ".join(rev))
            list_reviews.append(orig_rev)

    df = pd.DataFrame(list_reviews, columns=['text'])
    df['text'] = df['text'].str[2:]     # remove b'
    df['text'] = df['text'].str.strip() # remove white space
    df['polarity'] = indicator

    return df

def zero_padding(list_to_pad, max_length, pad_dimension):
    """
    This function takes a list and add list of zeros until max_length is reached.
    The number of zeroes in added list is determined by pad_dimension, which is the 
    same as the dimension of the word2vec model.

    This function is intended to handle one list only so it can be passed 
    into a dataframe as a lambda function.
    """
    # find number of padding vector needed
    num_pad = max_length - len(list_to_pad)

    # vector_pad = np.zeros(pad_dimension)
    vector_pad = np.asarray([0] * pad_dimension, dtype=np.float32)
    vector_pad = [vector_pad]    # convert to list of np.ndarray so we can append together 

    iteration = 0
    while iteration < num_pad:
        list_to_pad = np.append(list_to_pad, vector_pad, axis=0)
        iteration += 1
    
    return list_to_pad
    


if __name__ == "__main__":
    # load data into pandas df
    print('Creating dataset...')
    path_data = '../../data/movie_review'

    df_positive = create_dataset(path_data, 'positive')
    df_negative = create_dataset(path_data, 'negative')

    df = pd.concat([df_positive, df_negative]).reset_index(drop=True)

    # tokenization
    print('Tokenization...')
    df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

    # train word2vec
    print('Training word2vec...')


    train_model = False
    model_name = 'w2v_movie_review'
    emb_dim = 50

    min_count = 1

    if train_model:
        w2v = Word2Vec(df['text_token'].tolist(),
                    size=emb_dim,
                    window=5,
                    min_count=min_count,
                    negative=15,
                    iter=10,
                    workers=multiprocessing.cpu_count())
        w2v.save('../../model/{}.model'.format(model_name))
    else:
        w2v = Word2Vec.load('../../model/{}.model'.format(model_name))

    # embedding
    print('Embedding...')
    df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

    # padding 
    print('Padding...')
    # find max text length
    df['text_length'] = df['text'].apply(lambda x: len(x))
    max_length = max(df['text_length'])

    df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim))

    print(len(df['embedding'][1]))
    print(df['embedding'][1])
