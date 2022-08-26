# Just a quick pipeline file to compute things like number of words
import pandas as pd
import numpy as np
import re
#from tqdm import tqdm, tqdm_pandas

import pipeline_util as plu

# Change this list to change which conditionals get counted
cond_list = ["if","in_case","where","were","had","could","unless","should",
    "as_long_as","so_long_as","provided_that","otherwise","supposing"]

def compute_word_counts(pl):
    # There are actually two different possible word counts: the first is just
    # the number of tokens in the word-tokenized plaintext. This, to me, is not
    # what we want, since (for example) if there are a bunch of spaces between
    # each letter this method will produce some huge number of tokens, even
    # though each token is actually just one letter. So I think what we want to
    # do instead is count the number of words in the full_sentence values for
    # each contract.
    pl.iprint("compute_word_counts()")
    auth_df = pl.get_latest_auth_df()
    # First, while it's at the statement level, compute counts
    vars_to_keep = ["contract_id","article_num","sentence_num","statement_num",
                    "full_sentence","subnorm"]
    wc_df = auth_df[vars_to_keep].copy()
    wc_df["statement_count"] = 1
    # Subnorm-specific counts
    main_subnorms = pl.subnorm_list
    main_subnorms.remove("other")
    for cur_subnorm in main_subnorms:
        count_var = cur_subnorm + "_count"
        wc_df[count_var] = wc_df["subnorm"] == cur_subnorm
        wc_df[count_var] = wc_df[count_var].astype(int)
    # Aggregate to sentence level (full_sentence is redundant for different
    # statements within the same sentence)
    wc_groups = wc_df.groupby(['contract_id','article_num','sentence_num'])
    pl.iprint("Summing to sentence level")
    agg_dict = {'full_sentence':'first','statement_count':'sum','firm_count':'sum',
                'manager_count':'sum','union_count':'sum','worker_count':'sum'}
    sent_df = wc_groups.agg(agg_dict)
    # slow :(
    #main_subnorms = pl.subnorm_list
    #main_subnorms.remove("other")
    #main_subnorms = ["firm"]
    #for cur_subnorm in main_subnorms:
    #    cur_count_var = cur_subnorm + "_count"
    #    pl.iprint("Computing " + str(cur_count_var))
    #    sent_df[cur_count_var] = auth_groups["subnorm"].agg(lambda x:(x==cur_subnorm).sum())

    # Count the words. This regex method is faster than any other, according to
    # some stackoverflow post I'm too lazy to find again
    wordcount = re.compile(r'\w+')
    def num_tokens(sent_str):
        return len(wordcount.findall(sent_str))
    pl.iprint("Computing num_words")
    sent_df["num_words"] = sent_df["full_sentence"].apply(num_tokens)
    pl.iprint("Finished computing num_words")

    def count_conditionals(sent_str):
        # Uncomment this line if you want to produce a *vector* of counts, i.e.,
        # how many times each separate conditional appears in the string
        #count_vec = cur_text.count(cur_cond.replace("_"," ")) for cur_cond in cond_list]
        # Otherwise, we just sum all these individual counts up to get one final
        # conditional count
        return sum([sent_str.count(cur_cond.replace("_"," ")) for cur_cond in cond_list])
    sent_df["conditional_count"] = sent_df["full_sentence"].apply(count_conditionals)
    pl.iprint("Finished counting conditionals")

    # LIWC counts
    def num_matches(reg_str, test_str):
        # NaNs, by definition, have zero matches, but Python blows up if we don't
        # manually specify...
        if str(test_str) == "nan":
            return 0
        num_matches = len(re.findall(reg_str,test_str))
        return num_matches
    # Loop over each category specified by LIWC_DICT in pipeline.py
    for cur_liwc_cat in pl.LIWC_FPATHS:
        pl.iprint("Counting LIWC category: " + str(cur_liwc_cat))
        # Get the filename of the current LIWC file
        cur_liwc_fpath = pl.LIWC_FPATHS[cur_liwc_cat]
        cur_liwc_list = plu.stopwords_from_file(cur_liwc_fpath)
        cur_regex = plu.list_to_regex(cur_liwc_list)
        col_name = "liwc_" + str(cur_liwc_cat) + "_count"
        sent_df[col_name] = sent_df.apply(lambda row: num_matches(cur_regex, row["full_sentence"]), axis=1)

    #print(sent_df.head())
    #print(sent_df.columns)
    pl.iprint("Finished LIWC counts")

    # And aggregate to contract level
    article_df = sent_df.groupby(["contract_id","article_num"]).sum()
    contract_df = article_df.groupby(["contract_id"]).sum()
    contract_df.reset_index(inplace=True)

    output_csv_fpath = pl.get_sumstats_fpath(extension="csv")
    pl.iprint("Saving sumstats to " + output_csv_fpath)
    output_pkl_fpath = pl.get_sumstats_fpath(extension="pkl")
    plu.safe_to_csv(contract_df, output_csv_fpath, index=False)
    plu.safe_to_pickle(contract_df, output_pkl_fpath)
    output_dta_fpath = pl.get_sumstats_fpath(extension="dta")
    plu.safe_to_stata(contract_df, output_dta_fpath)

### This is the old count_conditionals, which makes a whole vector of counts rather
### than just a single aggregate count
# def count_conditionals(pl):
#     # Loops over each contract and just makes an overall count of each conditional word/phrase
#     # For now, just loads from the .txt files
#     all_counts = []
#     for cur_filename, cur_text in pl.streamFiles(eng_only=True):
#         print("Processing " + str(cur_filename))
#         # Convert the contract's text to all-lowercase
#         cur_text = cur_text.lower()
#         # Get the contract num
#         contract_num = int(cur_filename.split("_")[0])
#         count_vec = [contract_num] + [cur_text.count(cur_cond.replace("_"," ")) for cur_cond in cond_list]
#         all_counts.append(count_vec)
#     return all_counts

def summary_stats(pl):
    pl.iprint("summary_stats()")
    compute_word_counts(pl)
    # TODO: merge it back into auth_df to create <corpus>_sumstats.csv
