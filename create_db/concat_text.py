__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This script concatenate text in noteevent by HADM_ID.
This is needed because diagnoses, or the labels, are at HADM_ID level.
"""
import time
import sqlite3
import pandas as pd

if __name__ == "__main__":
      conn = sqlite3.connect('../../database/mimic.db')
      sql = 'SELECT a.hadm_id, a.text_cleaned as text ' \
            'FROM notes_cleaned a ' \
            'INNER JOIN (SELECT DISTINCT hadm_id '\
                        'FROM diagnoses_icd '\
                        'WHERE icd9_code in (\'4019\', \'42731\', \'4280\', \'51881\', \'5849\')) b ON '\
            'a.hadm_id = b.hadm_id' \
            ';'

      df = pd.read_sql_query(sql, conn)

      # concatenate strings
      df = df.groupby(['HADM_ID'], as_index=False).agg(' '.join)

      # write df to database
      df.to_sql('notes_concat', conn, if_exists='replace')
      print('df written to database.')
