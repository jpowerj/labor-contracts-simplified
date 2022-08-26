# An optional step you can add into the pipeline that just combines the split files
# created by compute_auth
import os

import pandas as pd
import numpy as np

import pipeline_util as plu

def combine_auth(pl):
    auth_path = pl.get_statement_auth_path()
    fpath_data = plu.sort_by_suffix(auth_path)
    new_fname = plu.remove_suffix(auth_path)
    auth_df = pd.DataFrame()
    for fnum, fpath in fpath_data:
        cur_df = pd.read_pickle(fpath)
        auth_df = pd.concat([auth_df,cur_df])
    # Once combined, save a version without a numeric suffix
    plu.safe_to_pickle(auth_df, os.path.join(pl.get_output_path(), new_fname))