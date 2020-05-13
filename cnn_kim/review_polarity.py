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

def create_dataset(path_data):
    """
    This function reads movie review data file and create pd.DataFrame.
    """
    list_reviews = []
    with open(os.path.join(path_data, 'rt-polarity.pos'), "rb") as f:
        for line in f:
            rev = []
            rev.append(str(line.strip()))

            orig_rev = clean_str(" ".join(rev))

            list_reviews.append(orig_rev)

    df_positive = pd.DataFrame(list_reviews, columns=['text'])
    df_positive['polarity'] = 1

    list_reviews = []
    with open(os.path.join(path_data, 'rt-polarity.neg'), "rb") as f:
            for line in f:
                rev = []
                rev.append(str(line.strip()))

                orig_rev = clean_str(" ".join(rev))

                list_reviews.append(orig_rev)

    df_negative = pd.DataFrame(list_reviews, columns=['text'])
    df_negative['polarity'] = 0

    return df_positive, df_negative


path_data = '../../data/movie_review'

df_positive, df_negative = create_dataset(path_data)

print(df_positive.head())
print(df_positive.shape)

print(df_negative.head())
print(df_negative.shape)
