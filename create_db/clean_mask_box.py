__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This module contains functions to clean de-identifying mask box 
for MIMIC-III text
"""
import time
import sqlite3
import pandas as pd

def clean_mask_box(df, name_text_col='TEXT', name_new_col='text_cleaned'):
    """
    This function removes mask box and create a new dataframe column.
    """
    # apply remov_box to all row in dataframe
    df[name_new_col] = df[name_text_col].apply(lambda x: remove_box(x))
    return df


def remove_box(text):
    """
    This function takes a text string and remove all the mask box insdie the text.
    """
    text_tmp = text
    for box in range(0, find_num_box(text)):
        text_tmp = remove_one_box(text_tmp)
    return text_tmp


def find_num_box(text):
    """
    This function finds number of mask boxes so we can iterate through the number
    to remove the mask box from given text.
    """
    # find index of beginning of mask box
    box_beg = []
    for i, char in enumerate(text):
        if char == '[':
            if text[i + 1:i + 3] == '**':
                box_beg.append(i)

    # find index of end of mask box
    box_end = []
    for j, char in enumerate(text):
        if char == '*':
            if text[j + 1:j + 3] == '*]':
                box_end.append(j)

    if len(box_beg) == len(box_end):
        #print(f'\n{len(box_beg)} of mas box(es) found!')
        return len(box_beg)
    else:
        print('Extra box element in text!')


def remove_one_box(text):
    """
    This function looks for first occurance of '[** **]', parse the text
    string and rejoin to remove the mask box.
    """
    index_beg = text.find('[**')
    index_end = text.find('**]') + 3

    if index_end > index_beg:
        str_pre = text[:index_beg]
        str_post = text[index_end:]
        return str_pre + str_post
    else:
        print('Mask box end appears before mask box beginning! Check text!')


if __name__ == "__main__":
    # load data
    conn = sqlite3.connect('../../database/mimic.db')
    sql = 'SELECT row_id, hadm_id, TEXT FROM noteevents'
    df = pd.read_sql_query(sql, conn)

    # clean mask box
    time0 = time.time()
    df = clean_mask_box(df, )
    # print(df['text'][0])
    time_elapsed = time.time() - time0
    task = 'Text cleaning'
    print('{} complete.    Total elapsed time: {}'.format(task, time_elapsed))

    # write df to database
    df.to_sql('notes_cleaned', conn, if_exists='replace')
    print('df written to database.')

    conn.close()
