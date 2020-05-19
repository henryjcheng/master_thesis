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
https://github.com/Shawn1993/cnn-text-classification-pytorch
"""
import os
import sys
import re
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
    with open(os.path.join(path_data, file_name), "rb") as f:
        for line in f:
            rev = []
            rev.append(str(line.strip()))

            orig_rev = clean_str(" ".join(rev))
            list_reviews.append(orig_rev)

    # assign 10-fold cross validation
    random.seed(1)
    list_rand = []
    for i in range(len(list_reviews)):
        list_rand.append(random.randint(1, 10))
    
    df = pd.DataFrame(zip(list_reviews, list_rand), columns=['text', 'fold'])
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
    # ===== 1. Preprocessing ===== 
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

    # ===== 2. Define CNN Architecture =====
    # Not sure how to write convolutional layer with custom filter/kernel size
    # so we will use the same kernel size as the filter, just to get the model working
    # looks like using bracket at kernel size let you define uneven kernels (x, y)
    # so the problem is then how to apply differnt kernel to the same input
    # and then bind them together as one input with different channels

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            kernel_size = [2, 3, 4, 5]
            self.conv1 = nn.Conv2d(1, 4, 50)      # input channel, output channel, kernel size
            self.pool = nn.MaxPool2d((217, 1), 1) 
            self.fc1 = nn.Linear(4 * 1 * 1, 1)
            self.sig = nn.Sigmoid()
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = x.view(-1, 4 * 4 * 1)
            x = self.fc1(x)
            x = sig(x)
            return x
    
    model = Net()

    # Define loss and optimization function
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print('GPU Device: {}'.format(device))

    model.to(device)

    # define training data
    # using 1 set of 10-fold
    train = df[df['fold'] != 10][['embedding', 'polarity']].reset_index(drop=True)
    test = df[df['fold'] == 10][['embedding', 'polarity']].reset_index(drop=True)

    epoch = 1
    # train 1 epoch
    running_loss = 0.0
    inputs, labels = train['embedding'].cuda(), train['polarity'].cuda()

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print(f'epoch: {epoch}')
    if False:
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch+1}, {i+1}] loss: {running_loss/2000}')
            running_loss = 0.0           
