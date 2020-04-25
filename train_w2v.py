__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This script trains word2vec model using MIMIC-III noteevets. 
MIMIC-III use a mask box to de-identify PHI, which needs to be removed first.

Reference: https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296 
"""
import sqlite3
import pandas as pd
from clean_mask_box import clean_mask_box

# load data
conn = sqlite3.connect('../database/mimic.db')
df = pd.read_sql_query("select TEXT from noteevents limit 5;", conn)
#print(df['TEXT'][0])

# clean mask box
df = clean_mask_box(df)
print(df['text'][0])
