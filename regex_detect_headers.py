# Quick code to *annotate* section headers via a simple regex (to match the BRAT
# annotation format)
import codecs
import glob
import os
import re
import unidecode

import pandas as pd
import numpy as np

# Match 1-9
DIGIT_WORDS = [r'ONE',r'TWO',r'THREE',r'FOUR',r'FIVE',r'SIX',r'SEVEN',r'EIGHT',r'NINE']
DIGIT_PAT = r'(?:' + r'|'.join(DIGIT_WORDS) + r')'
# Match 10-19
TENS_WORDS = [r'TEN',r'ELEVEN',r'TWELVE',r'THIRTEEN',r'FOURTEEN',r'FIFTEEN',r'SIXTEEN',
              r'SEVENTEEN',r'EIGHTEEN',r'NINETEEN']
TENS_PAT = r'(?:' + r'|'.join(TENS_WORDS) + r')'
# Match 20, 30, ..., 90
NUM_PREFIXES = [r'TWENTY',r'THIRTY',r'FORTY',r'FIFTY',r'SIXTY',r'SEVENTY',r'EIGHTY',r'NINETY']
PREFIX_PAT = r'(?:' + r'|'.join(NUM_PREFIXES) + r')'
# Match 21-29, 31-39, ..., 91-99
HYPHEN_PAT = r'(?:' + PREFIX_PAT + r'[- ]' + DIGIT_PAT + r')'
# Match all of the above
NUMWORD_PAT = r'(?:' + r'|'.join([DIGIT_PAT,TENS_PAT,PREFIX_PAT,HYPHEN_PAT]) + r')'

# Mangled S => need N at end. Otherwise SECT suffices
ART_PATS = r'(?:ARTICLE|SECTION|\w ?E ?C ?T ?\w ?\w ?N ?|S ?E ?C ?T ?[0-9a-zA-Z_ ]{1,6}|[0-9a-zA-Z_] ?R ?T ?\w? ?C ?L? ?E?|A ?r ?t ?\w ?c ?\w? ?e?|S ?e ?c ?t ?i? ?\w? ?n?)'

SIMPLE_REG = False
if SIMPLE_REG:
    # Simple version: no number-words
    ART_REG = re.compile(r'^.{0,3}(?:S ?E ?C ?T ?I? ?O? ?N? ?|A ?R ?T ?I ?C ?L? ?E?|A ?r ?t ?i ?c ?l? ?e?|S ?e ?c ?t ?i? ?o? ?an?)(?: |-)([0-9IiVvXxlOo -]{1,4}|[0-9]{1,2}[ABCD])(.+)?$', 
                         flags=re.M|re.UNICODE)
else:
    # Complicated version: number-words to pick up more headers (improves acc 
    # by like 3%, pretty important actually)
    ART_REG = re.compile(r'^.{0,3}' + ART_PATS + r'(?: |-)?([0-9IiVvXxlOo -]{1,4}|[0-9]{1,2}[ABCD]|' + NUMWORD_PAT + r')?(.+)?$', 
                         flags=re.M|re.UNICODE)

END_TOKENS = [r'in witness whereof', r'in witness wereof', r'whereof', r'wereof',
              r'signed this', r'dated at', r'for the company', r'for the union', r'effective date',
              r'signatories of', r'in witness thereof', r'thereof', r'signed at']
END_REG = re.compile(r'^(' + r'|'.join(END_TOKENS) + r')', flags=re.M|re.I|re.UNICODE)

APPENDIX_REG = re.compile(r'^(appendix|schedule)', flags=re.M|re.I|re.UNICODE)
# If set to one, scoring is more "strict" since we penalize for any chars in the
# gold standard anno that the computer anno didn't also mark (but see comment
# below for why 0 is good)
SCORE_PROPORTIONS = False

def overlap_len(range_start, range_end, guessed_annos, return_tuple=True):
    for iter_num, cur_guess in enumerate(guessed_annos):
        #print(f"{iter_num}: Checking overlap with {cur_guess}")
        gstart = cur_guess[0]
        gend = cur_guess[1]
        overlap_start = max(range_start, gstart)
        overlap_end = min(range_end, gend)
        overlap_num = overlap_end - overlap_start
        if overlap_num > 0:
            if return_tuple:
                return (overlap_num, cur_guess)
            # Otherwise we just return the overlap length
            return overlap_num
    # No matches
    if return_tuple:
        return (0, None)
    return 0
    # Old, slower I think
        #if overlap_num > 0:
            #print("overlap!")
            #print(f"human anno: {cur_ann_info}")
            #print(f"regex anno: {cur_guess}")
            #print(f"overlap = ({overlap_start},{overlap_end}), len {overlap_num}")

def compute_match_score(cur_row, art_annos, end_annos, return_series=False):
    #print(f"ann_info: {cur_row['ann_info']} ... ann_str: {cur_row['ann_str']}")
    cur_str = cur_row["ann_str"]
    # (Removing semicolons so there's always just spaces between the indices)
    cur_ann_info = str(cur_row["ann_info"]).replace(";"," ")
    range_elts = cur_ann_info.split()
    # First element should be the tag, rest should be char numbers
    ann_tag = range_elts[0]
    range_nums = [int(cur_num) for cur_num in range_elts[1:]]
    range_start = min(range_nums)
    range_end = max(range_nums)
    # Now get the overlap with the predicted annos, if any
    if ann_tag == "Person":
        # Check overlaps for the computer's guessed section heads
        overlap, which_overlap = overlap_len(range_start, range_end, art_annos, return_tuple=True)
        
    else:
        # Check for the computer's guessed end-of-contract tags
        overlap, which_overlap = overlap_len(range_start, range_end, end_annos)
    if SCORE_PROPORTIONS:
        # Hard mode: score based on *how much* of the gold standard annotation
        # the computer annotation marked
        overlap_score = overlap / len(cur_str)
    else:
        # Easy mode: just give it a score of 1 if it got some overlap (this is
        # totally good for our purposes since we really only care about this
        # for splitting up the contracts into more manageable parts)
        overlap_score = 1 if overlap > 0 else 0
    if return_series:
        return pd.Series([overlap_score, which_overlap], index=["match_score","which_overlap"])
    else:
        # We just return the score
        return pd.Series([overlap_score], index=["match_score"])

def detect_headers(ctext):
    # Replace these annoying unicode page end symbols with normal \n
    ctext = ctext.replace('\x0c','\n')
    
    # Debugging: uncomment this to see a slice of the doc
    #print(repr(ftext[2000:5000]))

    # Detect article heads
    art_results = ART_REG.finditer(ctext)
    art_annos = [(m.start(0), m.end(0), m[0]) for m in art_results]
    #print(art_annos)
    #print("-----")

    # Detect end of contract
    end_results = END_REG.finditer(ctext)
    end_annos = [(m.start(0), m.end(0), m[0]) for m in end_results]
    #print(end_annos)
    #print("-----")

    # Detect appendices/schedules [currently not used in evaluation -- see the args keyword
    # argument in the call to apply()]
    appendix_res = APPENDIX_REG.finditer(ctext)
    appendix_annos = [(m.start(0), m.end(0), m[0]) for m in appendix_res]
    #print(append_annos)

    return (art_annos, end_annos, appendix_annos)

def score_annotations(anno_fpath):
    anno_fname = os.path.basename(anno_fpath)
    #print(anno_fname)
    
    with codecs.open(anno_fpath, "r", "utf-8") as f:
        ftext = f.read()

    art_annos, end_annos, appendix_annos = detect_headers(ftext)

    # Put it all together
    # And evaluate these two given the gold standard
    # Load the gold standard
    gs_fpath = anno_fpath.replace(".txt",".ann")
    gs_df = pd.read_csv(gs_fpath, delimiter="\t", header=None, names=["id","ann_info","ann_str"])
    # Sometimes the files are empty since the doc actually didn't contain any
    # (readable) articles
    if len(gs_df) == 0:
        return (np.nan, None)
    gs_df["file"] = anno_fname
    
    #gs_df["match_score"] = gs_df.apply(compute_match_score, axis=1, args=(art_annos, end_annos))
    # sketchier but better way: have the apply'd function return a list
    match_df = gs_df.apply(compute_match_score, axis=1, args=(art_annos, end_annos), return_series=True)
    gs_df = pd.concat([gs_df,match_df],axis=1)
    #print(gs_df.head())
    #quit()
    avg_score = gs_df["match_score"].mean()
    #print(f"Avg score: {avg_score}")
    return (avg_score, gs_df)

def main():
    results_df = pd.DataFrame()
    avg_results = []
    contract_glob = "./annotations/BRAT_Annotations/*.txt"
    fpath_list = glob.glob(contract_glob)
    for cur_fpath in fpath_list:
        cur_fname = os.path.basename(cur_fpath)
        cur_score, anno_df = score_annotations(cur_fpath)
        #print(results_df.head())
        #print("*****")
        if anno_df is not None:
            #print(anno_df.head())
            results_df = pd.concat([results_df,anno_df], ignore_index=True, sort=False)
        avg_results.append((cur_fname, cur_score))
    print(results_df)
    print("-----")
    # Make a df of results for slightly prettier output
    avg_results_df = pd.DataFrame(avg_results, columns=["file","avg_score"])
    print(avg_results_df)
    print("-----")
    # Compute global avg
    just_scores = [res[1] for res in avg_results if not np.isnan(res[1])]
    overall_avg = sum(just_scores) / len(just_scores)
    print(f"Overall avg: {overall_avg}")
    print(f"Num gold standard annos: {len(results_df)}")

    ### Debugging zone
    #bad_file = "0257404a_eng.txt"
    #bad_file = "0045910a_eng.txt"
    #bad_file = "0175304a_eng.txt"
    #bad_df = results_df[results_df["file"] == bad_file].copy()
    #print(bad_df)
    #with codecs.open("./annotations/BRAT_Annotations/" + bad_file, "r") as f:
    #    bad_text = f.read()
    #print(bad_text[5000:6000])

if __name__ == "__main__":
    main()