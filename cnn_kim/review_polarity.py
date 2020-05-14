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
"""
import os
import sys
import re
import pandas as pd
import gensim.downloader as api
from nltk.tokenize import word_tokenize

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

            words = set(orig_rev.split())
            for word in words:
                if word is not None:
                    list_vocab.append(word)

    df = pd.DataFrame(list_reviews, columns=['text'])
    df['text'] = df['text'].str[2:]     # remove b'
    df['text'] = df['text'].str.strip() # remove white space
    df['polarity'] = indicator

    return df, list_vocab

if __name__ == "__main__":
    # load data into pandas df
    print('Creating dataset...')
    path_data = '../../data/movie_review'

    df_positive, vocab_positive = create_dataset(path_data, 'positive')
    df_negative, vocab_negative = create_dataset(path_data, 'negative')

    # load word2vec model by Google
    print('Load w2v model...')
    model = api.load("word2vec-google-news-300")

    # repalcing words not in vocab with 'unk'
    print('Replacing new words with \'unk\'...')
    vocab = []
    for word in vocab_positive:
        vocab.append(word)
    for word in vocab_negative:
        vocab.append(word)

    vocab_new = []
    for word in vocab:
        if word not in model.wv.vocab:
            vocab_new.append(word)
    
    list_positive = df_positive['text'].tolist()
    print(list_positive)

    # tokenization

    # embedding
    # df_positive['embedding'] = df_positive['text_token'].apply(lambda x: model[x])
    # df_negative['embedding'] = df_negative['text_token'].apply(lambda x: model[x])

    # print(df_positive['embedding'][:5])