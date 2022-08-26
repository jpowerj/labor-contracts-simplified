# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:19:34 2016

@author: elliott, Jeff J

Basically takes the raw statement dictionaries and converts them into a nice
human-readable csv
"""
# Python imports
from collections import Counter
import csv
import logging
import os
import shutil
import sqlite3

# 3rd party imports
import pandas as pd
from tqdm import tqdm

# Internal imports
import pipeline_util as plu
                        
other = ['agreement',
         'day',
         'assignment','leave','payment',
         'he/she','they','party']

conditionals = ['if','when','unless']
                #'where', 'whereas','whenever', 'provided that', 'in case']
    
def extract_pdata(pl, pdata_path):
    pl.iprint("Starting extract_pdata()")
    subcount = Counter()
    subnouncount = Counter()
    modalcount = Counter()
    # Loop over the items, getting counts but also producing a .csv where each
    # row is a statement
    if pl.batch_range:
        # Num to process is end - start + 1
        num_to_process = pl.batch_range[1] - pl.batch_range[0] + 1
    else:
        # Num to process is just num_contracts
        num_to_process = pl.compute_num_contracts()
    # Counting individual loop iterations
    iteration_num = 0
    # Counting the number of times we save and clear the accumulated data
    chunk_num = 0
    pdata_rows = []
    for statement_data in pl.pbar(pl.stream_parses(), total=num_to_process):
        contract_id = statement_data["contract_id"]
        # Loop over each statement, getting the subject/subject_branch/subject_tag
        subject = statement_data['subject']
        statement_dict = {'contract_id':contract_id,
            'article_num':statement_data['article_num'],
            'sentence_num':statement_data['sentence_num'],
            'statement_num':statement_data['statement_num'],
            'full_sentence':statement_data['full_sentence'],
            'full_statement':statement_data['full_statement'],
            'subject':statement_data['subject'], 'passive':statement_data['passive'],
            'subject_tags':statement_data['subject_tags'],
            'subject_branch':statement_data['subject_branch'],
            'object_tags':statement_data['object_tags'],
            'verb':statement_data['verb'], 'modal':statement_data['modal'],
            'md':statement_data['md'], 'neg':statement_data['neg'],
            'object_branches':statement_data['object_branches']}
        pdata_rows.append(statement_dict)
        subjectnouns = sorted([x for x,t in zip(statement_data['subject_branch'], statement_data['subject_tags']) if t.startswith('N')])
        subcount[subject] += 1
        if statement_data['md'] == 1:
            modalcount[subject] += 1
        for x in subjectnouns:
            if x != subject:
                    subnouncount[x] += 1
        iteration_num = iteration_num + 1
        # Print a message and save the statements every 100k
        if iteration_num % 100000 == 0:
            pl.iprint(f"Iteration {iteration_num}: Saving [Processing contract {contract_id}]")
            cur_df = pd.DataFrame(pdata_rows)
            pl.save_pdata_df(cur_df, chunk_num)
            chunk_num = chunk_num + 1
            pdata_rows.clear()
    # Make a Pandas df out of whatever's left in pdata_rows and save it
    cur_df = pd.DataFrame(pdata_rows)
    pl.save_pdata_df(cur_df, chunk_num)
    sub_counts_filename = os.path.join(pl.get_output_path(), 
                               pl.get_corpus_name() + "_subject_counts.pkl")
    pd.to_pickle(subcount,sub_counts_filename)
    modal_counts_filename = os.path.join(pl.get_output_path(), 
                                 pl.get_corpus_name() + "_modal_counts.pkl")
    pd.to_pickle(modalcount,modal_counts_filename)
    
    pl.iprint("Most common subjects:")
    pl.iprint(subcount.most_common()[:100])
    
def sum_by_contract(pl, allcsv_fpath, sumcsv_fpath):
    # Loads the all_statements.csv file produced by parseStatements() and sums
    # it to the contract level. This is an optional step, basically just here
    # in case we want to see contract-level stats about the parses
    df = plu.safe_read_csv(allcsv_fpath, level="statement")
    ## Everything should use contract_id now so we shouldn't need this line anymore
    #df.rename(index=str, columns={"contract_id":"contract_id"},inplace=True)
    df['count'] = 1
    grp_df = df.groupby('contract_id').sum().reset_index()
    grp_df.drop(["article_num","sentence_num","statement_num"],axis=1,inplace=True)
    plu.safe_to_csv(grp_df, sumcsv_fpath)

def get_parse_data(pl, pdata_path=None):
    pl.iprint("Starting get_parse_data()")
    if pdata_path is None:
        pdata_path = pl.get_pdata_path()
    extract_pdata(pl, pdata_path)
    pl.iprint("Finished get_parse_data()")
