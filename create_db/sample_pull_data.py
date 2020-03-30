"""
This module contains codes to pull data from mimic.db created by create_db.sh.
1. test that the mimic.db is functional
2. documents code
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('../../database/mimic.db')
df = pd.read_sql_query("select * from d_cpt limit 5;", conn)
print(df)