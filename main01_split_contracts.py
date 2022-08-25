# Python imports
import glob
# For saving the article lists as .json
import json
import os

# 3rd party imports
# For saving the article lists as .pkl
import joblib
# For np.random.choice() sampling
import numpy as np

# Relative imports
import regex_detect_headers as rdh
import detect_sections_elliott as dse

def drop_article(art_data):
    # A "post-processing" step, which just removes articles that are overly
    # gibberish-filled
    if "........" in art_data['text']:
        return True
    return False

def _get_fpath_elts(fpath):
    # Gets id and lang from filename
    base = os.path.basename(fpath).split(".")[0]
    return base.split("_")

def get_id(fpath):
    # Helper function, gets the id from the filename
    elts = _get_fpath_elts(fpath)
    return elts[0]

def get_lang(fpath):
    elts = _get_fpath_elts(fpath)
    return elts[1]

def load_plaintext(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        doc_text = f.read()
    return doc_text

def _make_dirs(fpath):
    # Helper function that ensures the directories exist so you
    # can actually save the file
    # First convert the (potentially) relative path to absolute
    abs_fpath = os.path.abspath(fpath)
    # Now call makedirs on it
    dirname = os.path.dirname(abs_fpath)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def safe_to_json(data, fpath):
    _make_dirs(fpath)
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def safe_to_pickle(data, fpath):
    _make_dirs(fpath)
    # Now we should be able to save it without error (even if
    # the directory/directories didn't exist before)
    joblib.dump(data, fpath)

def split_contract(contract_fpath, splitter):
    # Splits the articles in the contract at fpath, returning a dict
    # containing 'header', 'text', 'section_num', 'contract_id', and 'lang'
    # Load the plaintext from the fpath
    doc_text = load_plaintext(contract_fpath)
    doc_id = get_id(contract_fpath)
    doc_lang = get_lang(contract_fpath)
    if splitter == "regex":
        # Get the regex split annotations
        art_annos, end_annos, appendix_annos = rdh.detect_headers(doc_text)
        # Do the actual splitting
        art_list = split_on_annos(doc_text, art_annos, doc_id, doc_lang)
        return art_list
    else:
        # Use Elliott's article-splitting code
        arts, headers = dse.detect_sections(doc_text)
        # Convert to the dict format for compatibility with regex splitter
        # It looks like len(headers) is almost always greater than (often
        # like double or triple) len(arts). So for now I'm ignoring headers
        art_list = []
        for i in range(len(arts)):
            cur_art = arts[i]
            cur_art_data = {'header': None, 'text': cur_art, 'section_num': i,
                        'contract_id': doc_id, 'lang': doc_lang}
            art_list.append(cur_art_data)
        return art_list
    
def split_contracts(pl):
    # Get the paths to the folders where the pkl and json files should be saved
    pkl_path = os.path.join(pl.get_output_path(), f"01_artsplit_{pl.splitter}_pkl")
    json_path = os.path.join(pl.get_output_path(), f"01_artsplit_{pl.splitter}_json")
    # Now loop over the fpaths, splitting articles for each
    fpaths = pl.plaintext_fpaths
    for fnum, cur_fpath in enumerate(fpaths):
        doc_id = get_id(cur_fpath)
        if fnum % 100 == 0:
            print(f"#{fnum}: Splitting articles for {doc_id} using splitter {pl.splitter}")
        art_list = split_contract(cur_fpath, pl.splitter)
        # And save the article list as .pkl (for internal use) and .json
        # (for human reading)
        pkl_fpath = os.path.join(pkl_path, doc_id + ".pkl")
        safe_to_pickle(art_list, pkl_fpath)
        json_fpath = os.path.join(json_path, doc_id + ".json")
        safe_to_json(art_list, json_fpath)
    print(f"artsplit pkls saved to {pkl_path}")
    print(f"artsplit jsons saved to {json_path}")

def split_on_annos(doc_text, art_annos, contract_id, lang):
    # Take the article annotations and use them to split the text
    #print(art_annos)
    just_ranges = [(a[0],a[1]) for a in art_annos]
    #print("anno ranges:")
    #print(just_ranges)
    #print(len(just_ranges))
    gap_ranges = [(i[1],j[0]) for i,j in zip(art_annos, art_annos[1:])]
    #print("gap_ranges:")
    #print(gap_ranges)
    #print(len(gap_ranges))
    # Cool, now we pair them up as (header, text)
    pairs = [(just_ranges[i],gap_ranges[i]) for i in range(len(gap_ranges))]
    #print("pairs")
    #print(pairs)
    #print(len(pairs))
    # And finally we get the actual string ranges
    data = [{'header': doc_text[h[0]:h[1]], 'text': doc_text[t[0]:t[1]]} for h,t in pairs]
    final_data = [d for d in data if not drop_article(d)]
    # Now add in the section_num, contract_id, and lang
    for i in range(len(final_data)):
        final_data[i]['section_num'] = i
        final_data[i]['contract_id'] = contract_id
        final_data[i]['lang'] = lang
    return final_data