# This used to be just a function within main04_compute_auth.py, but I moved it
# to its own file so that you can run it separately if need be (i.e., skip the
# re-computation of auth measures but change the summing function)
import pandas as pd
import numpy as np
import pipeline_util as plu

def sum_auth(pl):
    """ Takes the authority measures computed by compute_auth.py and sums them
    to the contract level

    Requires: 
    """
    #breakpoint()
    pl.iprint("sum_auth()")
    # Loads the df chunks in 04_<corpus>_auth folder produced by compute_auth and sums
    # them together to the contract level
    summed_df = None
    for chunk_num, cur_chunk_df in pl.stream_statement_auth():
        # Sum the current chunk
        summed_chunk_df = sum_auth_df(pl, cur_chunk_df)
        # And concatenate it to the full-data df
        if summed_df is None:
            summed_df = summed_chunk_df
        else:
            summed_df = pd.concat([summed_df, summed_chunk_df])
    # Now we do a final sum to obtain the (non-chunked) contract-level dataset
    final_summed_df = summed_df.groupby(["contract_id"]).sum()
    final_summed_df.reset_index(inplace=True)
    pl.save_authsums_df(summed_df)
    
    sum_csv_fpath = pl.get_authsums_fpath(extension="csv")
    sum_df.to_csv(sum_csv_fpath, index=False)
    sum_pkl_fpath = pl.get_authsums_fpath(extension="pkl")
    sum_df.to_pickle(sum_pkl_fpath)
    pl.iprint("Saved " + str(sum_csv_fpath))

def sum_auth_df(pl, auth_df):
    """
    Helper function for sum_auth(), which sums up a *single* (chunk) dataframe containing
    auth measures, so that they can be combined into one final summed df by sum_auth()
    """
    unique_ids = ["contract_id","subnorm"]
    grp_df = auth_df.groupby(unique_ids).sum()
    grp_df.reset_index(inplace=True)
    def conditional_measure(x,subnorm,measure):
        # This function returns the actual measure if the subnorm of the row
        # is equal to the subnorm argument, and 0.0 otherwise. This makes it
        # so that when we "sum" up to just contract_id we get just the measure
        # for the subnorm of interest
        return x[measure] if x["subnorm"]==subnorm else 0.0
    subnorms_to_process = pl.subnorm_list
    if "other" in subnorms_to_process:
        subnorms_to_process.remove("other")
    for cur_measure in pl.AUTH_MEASURES:
        pl.iprint("Making cols for " + cur_measure)
        for cur_subnorm in subnorms_to_process:
            print("Making cols for " + cur_measure + " x " + cur_subnorm)
            new_col_name = cur_measure + "_" + cur_subnorm
            grp_df[new_col_name] = grp_df.apply(conditional_measure,
                axis="columns",args=(cur_subnorm,cur_measure))
    # And now just sum! It's not actually summing anything, just getting rid
    # of all the 0.0 cells, for example permission_worker in a row with subnorm firm
    sum_df = grp_df.groupby(["contract_id"]).sum()
    sum_df.reset_index(inplace=True)

    # Return it so these summed chunk dfs can be combined into a final summed df
    # by sum_auth()
    return sum_df

#def main():
#    print("Only for testing!")
#
#if __name__ == "__main__":
#    main()