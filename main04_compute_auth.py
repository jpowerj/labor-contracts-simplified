# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:19:34 2016

@author: elliott
"""
# Python imports
import csv
import logging
import os
import pickle
import sqlite3
import sys

# 3rd party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# "Internal" imports
import pipeline_util as plu

## These will be removed in a future version -- replaced by the Pipeline class
#from global_functions import (debugPrint, streamStatements)
#from global_vars import (data_db_name, collection_name, CORPUS_DICT, CORPUS_NAME,
#    SUBJECT_NORM, USE_MONGO)

subdict = {'other':0,'worker':1,'union':2,'firm':3,'manager':4,
           0:'other',1:'worker',2:'union',3:'firm',4:'manager'}

snpdata_header = ["contract_id","section_num","sentence_num","statement_num",
                      "md","strict_modal","neg","passive","verb","object_branches",
                      "full_sentence"]

def check_strict_modal(statement_row):
    strict_modal = False
    if statement_row['md']:
        strict_modal = statement_row['modal'] in ['shall','must','will']
    return strict_modal

def check_neg(statement_row):
    return statement_row['neg'] == 'not'

## This is inside a dict comprehension now
#def get_subnorm_rows(subnorm, df):
#    # Takes the full df and makes a sub-df of just the subnorm rows
#    subnorm_df = df.loc[df["subnorm"] == subnorm]
#    return subnorm_df

def compute_statement_auth(pl, chunk_num, pdata_chunk_df):
    pl.iprint(f"starting compute_statement_auth()")
    # Uncomment below if the corpus is small enough to be processed all at once
    #pdata_df = pl.load_pdata_df()
    ### Add some initial columns to the df that will help us when we compute
    ### auth measures below
    vars_to_keep = ["contract_id","article_num","sentence_num","statement_num",
                    "subject","md","verb","passive","full_sentence"]
    ### Soo after this we're grabbing stuff out of auth_df A LOT.
    ### so I'm renaming it to just df
    df = pdata_chunk_df[vars_to_keep].copy()
    # We can save memory by converting some of the ints to bools
    df["md"] = df["md"].astype('bool')
    df["passive"] = df["passive"].astype('bool')
    df["subject"] = df["subject"].str.lower()
    df["subnorm"] = df["subject"].apply(pl.normalize_subject)
    # Strict modal check. axis=1 means apply row-by-row
    df["strict_modal"] = pdata_chunk_df.apply(check_strict_modal,axis=1).astype('bool')
    df["neg"] = pdata_chunk_df['neg'].apply(lambda x: x == 'not').astype('bool')

    with tqdm(total=13) as pbar:
        df['count'] = 1
        # permissive modals are may and can
        df['permissive_modal'] = (df['md'] & ~df['strict_modal']).astype('bool')
        pbar.update(1)

        # obligation verbs 
        df_passive = df['passive']
        df['obligation_verb'] = (df_passive & 
                              df['verb'].isin(['require', 'expect', 'compel', 'oblige', 'obligate'])).astype('bool')
        pbar.update(1)

        # constraint verbs 
        df['constraint_verb'] = (df_passive & 
                              df['verb'].isin(['prohibit', 'forbid', 'ban', 'bar', 'restrict', 'proscribe'])).astype('bool')
        pbar.update(1)
      
        # permissiion verbs are be allowed, be permitted, and be authorized
        df['permission_verb'] =  (df_passive &  
                               df['verb'].isin(['allow', 'permit', 'authorize'])).astype('bool')
        pbar.update(1)
      
        df_notpassive = ~df_passive
        df['entitlement_verb'] =  (df_notpassive &  
                                 df['verb'].isin(['have', 'receive','retain'])).astype('bool')
        pbar.update(1)
      
        df['promise_verb'] = (df_notpassive & 
                          df['verb'].isin(['agree','promise','commit','recognize',
                                      'consent','assent','affirm','assure',
                                      'guarantee','insure','ensure','stipulate',
                                      'undertake','pledge'])).astype('bool')
        pbar.update(1)

        pl.dprint("Computed up to promise_verb")

        df['special_verb'] = (df['obligation_verb'] | df['constraint_verb'] | df['permission_verb'] | df['entitlement_verb'] | df['promise_verb']).astype('bool')
        pbar.update(1)
      
      
        df['active_verb'] = (df_notpassive & ~df['special_verb']).astype('bool')
        pbar.update(1)
      
        #df['verb_type'] = 0 + 1 *df['passive'] + 2*df['obligation_verb'] + 3*df['constraint_verb'] + 4*df['permission_verb'] + 5*df['entitlement_verb']
         
        df_neg = df['neg']
        df_notneg = ~df_neg
        df['obligation'] = ((df_notneg & df['strict_modal'] & df['active_verb']) |     #positive, strict modal, action verb
                            (df_notneg & df['strict_modal'] & df['obligation_verb']) | #positive, strict modal, obligation verb
                            (df_notneg & ~df['md'] & df['obligation_verb'])).astype('bool')           #positive, non-modal, obligation verb
        pbar.update(1)
      
        df['constraint'] = ((df_neg & df['md'] & ~df['obligation_verb']) | # negative, any modal, any verb except obligation verb
                            (df_notneg & df['strict_modal'] & df['constraint_verb'])).astype('bool') # positive, strict modal, constraint verb
        pbar.update(1)
                
        df['permission'] = ((df_notneg & ( (df['permissive_modal'] & df['active_verb']) | 
                          df['permission_verb'])) | 
                          (df['neg'] & df['constraint_verb'])).astype('bool')
        pbar.update(1)
                          
                          
        df['entitlement'] = ((df_notneg & df['entitlement_verb']) |
                          (df_notneg & df['strict_modal'] & df['passive']) |
                          (df_neg & df['obligation_verb'])).astype('bool')
        pbar.update(1)
    pl.iprint("Authority measures computed.")
    # Testing sqlite3 saving b/c nothing else is working :(
    #con = sqlite3.connect("../canadian_output/canadian_stauth.sqlite")
    #df.to_sql("statement_auth", con, if_exists="replace")
    #con.close()
    # 5th attempt failed. Just gonna do it in batches :/
    pl.save_statement_auth(df, chunk_num)

    #grp.replace(np.nan,0,inplace=True)
    #pickle_filename = output_prefix + "-excerpt.pkl"   
    #grp.to_pickle(pickle_filename)
    
    #df.to_pickle('df-contract-1.pkl')

def compute_auth(pl):
    # Compute the statement-level authority measures, in batches since the full
    # thing crashes on TL :/
    # Stream over the 03_<corpus>_pdata_<chunk>.csv files
    for chunk_num, cur_chunk_df in pl.stream_pdata():
        compute_statement_auth(pl, chunk_num, cur_chunk_df)
    # TODO: compute the *normalized* auth measures as well before returning
    pl.iprint("About to return from compute_auth()")
