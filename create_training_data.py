__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This script creates training dataset for CNN using concatenated notes and diagnoses.

Input format:

from notes_concat: 
      HADM_ID                                       text
    0  100001  Admission Date:        Discharge Date:...
    1  100003  Admission Date:        Discharge Date:...
    2  100006  Admission Date: Discharge Date:  \n\nD...

from diagnoses_icd:
       HADM_ID ICD9_CODE
    0   163353     V3001
    1   163353      V053
    2   163353      V290

Output format:

"""
import sqlite3
import pandas as pd

# load data
conn = sqlite3.connect('../database/mimic.db')
sql = 'SELECT hadm_id, text ' \
      'FROM notes_concat' \
      ';'
df = pd.read_sql_query(sql, conn)

sql = 'SELECT hadm_id, icd9_code ' \
      'FROM diagnoses_icd ' \
      'WHERE icd9_code in (\'4019\', \'42731\', \'4280\', \'51881\', \'5849\')' \
      ';'
df_diag = pd.read_sql_query(sql, conn)

# turn icd9_code into list by concatenate into string first
# then turn the string to list
df_diag = df_diag.groupby('HADM_ID', as_index=False).agg(','.join)
df_diag['ICD9_CODE'] = df_diag['ICD9_CODE'].str.split(',')
