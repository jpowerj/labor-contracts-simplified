"""
The main file that holds the Pipeline class, which is used to initialize,
set global settings, and run the pipeline.
"""

# Python imports
import codecs
from collections import OrderedDict
import datetime
import glob
import importlib
import inspect
import logging
import os
import random
import sqlalchemy
import sys

# 3rd party imports
import gensim
import joblib
from nltk.corpus import stopwords
import pandas as pd
from pymongo import MongoClient
import numpy as np
from pathlib import Path
from tqdm import tqdm
from unidecode import unidecode

# Pipeline imports (lda and meta stuff can be imported in an if
# statement in the constructor if do_lda and/or do_meta are true)
import pipeline_util as plu

class Pipeline(object):

#######################
####### GLOBAL VARS ###
#######################

    AUTH_MEASURES = ["obligation","constraint","permission","entitlement"]
    CANSIM_DICT_FPATH = os.path.join(".","cansim_data","cansim_naics_dic.csv")
    CANSIM_FPATH = os.path.join(".","cansim_data","cansim_year_prov_ind.csv")
    CODEBOOK_FPATH = os.path.join(".","codebook.csv")
    DEFAULT_LANG = "eng"
    # Path to the elections data
    ELECTION_DATA_PATH = os.path.join(".","elections")
    # csv containing the old id -> new id mapping (shouldn't be needed after everything
    # is re-run!)
    ID_CROSSWALK_FPATH = os.path.join("contract_filenames.csv")
    # Path to inflation dataset
    INFLATION_FPATH = os.path.join("..","canadian_data","canadian_cpi.csv")
    # Path to the Lextek stopword list file.
    LEXTEK_FPATH = os.path.join(".","word_lists", "lextek_contracts.txt")
    # Can customize which LIWC categories you want to include by changing this
    # Path to the folder containing the LIWC category-by-category lists
    LIWC_PATH = os.path.join(".","word_lists","liwc")
    LIWC_FPATHS = {"number": os.path.join(LIWC_PATH, "024-number.txt"),
                   "quant": os.path.join(LIWC_PATH, "025-quant.txt"),
                   "certainty": os.path.join(LIWC_PATH, "055-certain.txt"),
                   "tentative": os.path.join(LIWC_PATH, "054-tentat.txt")}
    # First in the list is the default OCR engine
    OCR_OPTIONS = ["abbyy","adobe","tesseract"]
    # First in the list is the default parser
    PARSER_OPTIONS = ["spacy2","spacy1","corenlp","parsey"]
    # Path to directory containing *this* (pipeline.py) file
    PIPELINE_PATH = os.path.dirname(os.path.abspath(__file__))
    # Path to excel file with strike data
    STRIKE_FPATH = os.path.join("..","canadian_data","wsrs_stoppages-20160721.xlsx")

##############################
####### INSTANCE FUNCTIONS ###
##############################
    # Actually goes and counts the number of plaintext files
    def compute_num_contracts(self):
        if self.use_mongo:
            client = MongoClient()
            db = client["contracts"]
            collection = db[self.get_corpus_name()]
            return collection.count_documents({})
        else:
            return len(self.get_plaintext_fpaths())
    
    # This is how many possible digits the contract files can have
    # (so that they are correctly padded, so lexical sort = numeric sort)
    # Update: sigh. This can't just be len(str(num_contracts)), since
    # for example canadian_sample has 5 digits but only 100 contracts...
    def compute_num_digits(self):
        # Basically need to go over filename list and find the "widest" name
        ## This commented-out code is slow and only for debugging
        #length_map = {cur_name:len(cur_name) for cur_name in self.get_plaintext_filenames()}
        #print(length_map)
        #quit()
        cid_length = lambda fname: len(str(plu.get_contract_id(fname)))
        # Check if plaintext path is empty (happens if we're on AWS)
        filenames = self.get_plaintext_filenames()
        if not filenames:
            # Check for the sections pkls instead
            filenames = self.get_article_filenames()
        #print("First filename: " + cid_length(filenames[0]))
        #print("Last filename: " + filenames[-1])
        name_lengths = [cid_length(cur_name) for cur_name in filenames]
        max_length = max(name_lengths)
        return max_length

    # Prints to console/log iff debug==True
    def dprint(self, msg):
        if self.debug:
            # Find out which file called it
            cur_frame = inspect.currentframe()
            outer_frames = inspect.getouterframes(cur_frame, 2)
            # It's really the filepath, so get just the basename
            caller = os.path.basename(outer_frames[1].filename)
            if self.use_logging:
                logging.debug(msg)
            else:
                print("[DEBUG: " + str(caller) + "] " + str(msg))

    def get_pipeline_summary(self):
        pl_data = []
        pl_data.append("Pipeline created for corpus: " + self.get_corpus_name())
        if self.sample_N:
            pl_data.append("--> Sample of size " + str(self.sample_N))
            pl_data.append("Sampled contracts: " + str(self.get_contract_ids()))
        output_txt = "\n".join(pl_data)
        return output_txt

    def gen_filename(self, contract_id, contract_lang, file_extension):
        # Figures out the "correct" file prefix by padding with zeros
        # and appending the language
        #logging.debug("gen_filename(): self.num_digits = " + str(self.num_digits))
        filename = contract_id + "_" + contract_lang + "." + file_extension
        #logging.debug("final filename: " + filename)
        return filename

    def gen_stopwords(self, print_stopwords=False):
        """ This used to be in pipeline_util.py, but moved here since it is
        actually run-specific (since the subjectnorm keys are themselves used
        as stopwords). The final list of stopwords is the union of NLTK 
        stopwords, stopwords from lextek, and list of subjects """
        nltk_stoplist = set(stopwords.words('english'))
        if print_stopwords:
            print("NLTK stoplist: " + str(sorted(list(nltk_stoplist))))
        lextek_stoplist = set(plu.stopwords_from_file(self.LEXTEK_FPATH))
        if print_stopwords:
            print("Lextek stoplist: " + str(sorted(list(lextek_stoplist))))
        subject_list = set(self.subject_list)
        if print_stopwords:
            print("Filtered subjects: " + str(sorted(list(subject_list))))
        word_set = nltk_stoplist.union(lextek_stoplist).union(subject_list)
        return list(word_set)

    def get_article_filenames(self):
        article_path = self.get_artsplit_path()
        all_fpaths_glob = os.path.join(article_path, "*.pkl")
        all_fpaths = glob.glob(all_fpaths_glob)
        all_fpaths.sort()
        return all_fpaths

    def get_contract_ids(self, include_lang=False):
        """
        Returns a list of the contract_ids for each contract in the corpus

        include_lang: For debugging, returns an (id, lang) tuple
        """
        if self.use_mongo:
            client = MongoClient()
            db = client["contracts"]
            collection = db[self.get_corpus_name()]
            all_contracts = collection.find({'lang':{'$in':self.lang_list}})
            if include_lang:
                all_items = [(c["_id"],c["lang"]) for c in all_contracts]
            else:
                all_items = [c["_id"] for c in all_contracts]
            # debugging
            #print([lang for lang in collection.find({'lang':{'$in':self.lang_list}}).distinct('lang')])
            return all_items
        else:
            return [plu.get_contract_id(fname) for fname in self.get_plaintext_filenames()]

    def get_plaintext_filenames(self):
        all_paths = self.get_plaintext_fpaths()
        just_names = [os.path.basename(cur_path) for cur_path in all_paths]
        return just_names

    def get_plaintext_fpaths(self):
        """ Returns a list of all plaintext files (except files outside of the
        languages in self.lang_list). Importantly, it uses glob's recursive
        setting to find *all* .txt files within the corpus directory by default,
        so you need to set the recurse_corpus_dir constructor param to False if
        you don't want .txt files within subfolders to be included. """
        #breakpoint()
        self.dprint("get_plaintext_fpaths()")
        plaintext_path = self.get_plaintext_path()
        self.dprint("Using plaintext_path " + str(plaintext_path))
        #self.dprint("corp_extensions: " + str(self.corp_extensions))
        all_fpaths = []
        for cur_extension in self.corp_extensions:
            if not self.recurse_corpus_dir:
                # Just look in the main directory
                glob_str = "*." + cur_extension
            else:
                # Look in all subdirectories of the corpus directory as well
                glob_str = "**/*." + cur_extension
            #self.dprint("Using glob_str " + glob_str)
            #self.dprint("Using plaintext_path: " + str(plaintext_path))
            glob_results = sorted(Path(plaintext_path).glob(glob_str))
            # Annoyingly, need to convert these to strings
            glob_strs = [str(cur_result) for cur_result in glob_results]
            all_fpaths.extend(glob_strs)
        if len(all_fpaths) == 0:
            raise Exception(f"No files found with extensions {self.corp_extensions}")
        #self.dprint("all_fpaths: " + str(all_fpaths))
        self.dprint(f"inferred lang: {plu.get_contract_lang(all_fpaths[0])}")
        lang_fpaths = [cur_fpath for cur_fpath in all_fpaths 
                if any(cur_lang in plu.get_contract_lang(cur_fpath) for cur_lang in self.lang_list)
                or len(self.lang_list)==0]
        # SORT (by contract num, implicit in the filenames) before returning!
        # (This means that if the num_digits is messed up, the iterator order
        # will also be messed up)
        lang_fpaths.sort()
        return lang_fpaths

    # Just a convenience function, so I don't mess up the sample path/files
    def get_sample_corpus_name(self):
        if not self.sample_N:
            msg1 = "Pipeline is not set to run on a sample"
            msg2 = "(so this function shouldn't be called)"
            raise Exception(msg1 + " " + msg2)
        return self.corpus_name + "_sample_" + str(self.sample_N)

    def get_sample_corpus_path(self):
        if not self.sample_N:
            msg1 = "Pipeline is not set to run on a sample"
            msg2 = "(so this function shouldn't be called)"
            raise Exception(msg1 + " " + msg2)
        return os.path.join(self.corpus_path,"..",self.get_sample_corpus_name())

    # Prints msg to console/log (whether or not debug==True).
    def iprint(self, msg):
        if self.use_logging:
            logging.info(msg)
        else:
            cur_frame = inspect.currentframe()
            outer_frames = inspect.getouterframes(cur_frame, 2)
            # It's really the filepath, so get just the basename
            caller = os.path.basename(outer_frames[1].filename)
            print("[INFO: " + str(caller) + "] " + str(msg))

    def load_codebook(self):
        """ Loads codebook.csv and converts it into a variable->label dict """
        codebook_df = pd.read_csv(self.CODEBOOK_FPATH)
        codebook_dict = {row["var_name"]:row["var_desc"] for row_num, row in codebook_df.iterrows()}
        return codebook_dict

    def log_time(self, event):
        cur_timestamp = '{:%Y-%m-%d_%H.%M.%S}'.format(datetime.datetime.now())
        if not hasattr(self, 'tlog_fpath'):
            # Create the timing log file
            tlog_filename = f"timing_log_{cur_timestamp}.txt"
            self.tlog_fpath = os.path.join("logs",tlog_filename)
            plu.safe_append_to_file(self.get_pipeline_summary(), self.tlog_fpath)
        plu.safe_append_to_file(event + ": " + cur_timestamp, self.tlog_fpath)

    def normalize_subject(self, subject):
        # It would be nice if we could just use subnorm_map on its own rather
        # than a function, but we need to transform to "other" if it's not a
        # key in subnorm_map, so alas, an if statement
        # (Also, we lowercase the subject before checking. Learned the hard way)
        # Get the string representation in case its a weird format (like NaN)
        subject = str(subject)
        # Make sure it's in normal Python-friendly unicode
        subject = unidecode(subject)
        # Lowercase
        subject = subject.lower()
        # Transform plural to singular (if plural)
        subject = plu.get_singular(subject)
        if subject in self.subnorm_map:
            return self.subnorm_map[subject]
        else:
            return "other"

    def num_to_id(contract_num):
        """ Converts the (old) contract_num identifiers to the (new) pdf-filename-based
        identifiers. Should only be relevant until we transition everything to the new IDs!
        (i.e., we should be able to delete this after a while) """
        # The old id var is contract_num. The new id var is the file_name column but *without*
        # the .pdf extension
        if self.id_df is None:
            self.id_df = pd.read_csv(self.ID_CROSSWALK_FPATH)
            self.id_df["contract_id"] = self.id_df["file_name"].str.replace(".pdf","")
        return self.id_df.loc[self.id_df["contract_num"] == contract_num,"contract_id"].values[0]

    def pbar(self, iter_object, **kwargs):
    	if self.progress_bars:
    		return tqdm(iter_object, **kwargs)
    	else:
    		return iter_object

    def preprocess_text(self, contract_str):
        """ Uses gen_stopwords() and gensim's preprocess_string() method to perform
        the full preprocessing pipeline on contract_str.

        Added 2019-03-07 by JJ """
        # First we should remove our custom stopwords
        custom_stopwords = self.gen_stopwords()
        filtered_str = " ".join(w for w in contract_str.split() if w not in custom_stopwords)
        # Now use gensim's parsing library
        final_doc = gensim.parsing.preprocess_string(filtered_str)
        return final_doc

    #def save_artsplit(self, artsplit_data):
    #    # Put the current article list into Mongo
    #    client = MongoClient()
    #    db = client["contracts"]
    #    collection = db["canadian"]
    #    contract_id = artsplit_data["contract_id"]
    #    articles = artsplit_data[""]
    #    collection.update_one({'_id':contract_id},{'$set':{'articles':}})

    def save_artsplit_contract(self, contract_data, pkl_debug=False):
        # Make sure the directory exists
        artsplit_path = self.get_artsplit_path()
        plu.safe_open_dir(artsplit_path, is_fpath=False)
        contract_fpath = os.path.join(artsplit_path, contract_data["contract_id"] + ".json")
        if pkl_debug:
            self.iprint("Saving split contract to " + str(contract_fpath))
        plu.json_dump(contract_data, contract_fpath)

    def save_parsed_statements(self, statement_list):
        """
        Since the parallelism requires us to process all of the statements in all of the
        contracts in a giant batch (that it internally optimizes), at the end we need to
        save a big .pkl file containing all the parse data here.
        """
        parses_fpath = self.get_parses_fpath()
        joblib.dump(statement_list, parses_fpath)

    def save_pdata_rows(self, statement_list):
        # Make a df out of the list
        statement_df = pd.DataFrame(statement_list)
        # Save it to the sqlite db, making sure to set if_exists="append"
        engine = sqlalchemy.create_engine('sqlite:///../canadian_output/canadian.sqlite')
        # Sigh. Need to take any variable that is of type list and convert it to a string
        #print(statement_df.columns)
        statement_df["subject_tags"] = statement_df["subject_tags"].apply(plu.combine_list)
        statement_df["subject_branch"] = statement_df["subject_branch"].apply(plu.combine_list_of_lists)
        statement_df["object_tags"] = statement_df["object_tags"].apply(plu.combine_list_of_lists)
        statement_df["object_branches"] = statement_df["object_branches"].apply(plu.combine_list_of_lists)
        statement_df.to_sql("pdata", engine, if_exists="append", index=False)

    def save_timing_log(self):
        # Output timing log in csv format. Vars are section (i.e., step in the
        # pipeline), start_time in Python datetime format, finish_time, duration
        # which is just finish_time - start_time, rate_hr = # contracts/duration
        # in hours, rate_min = # contracts/duration in minutes
        #csv_rows = [["section","start_time","finish_time","duration_hr",
        #             "duration_min","rate_hr","rate_min"]]
        section_dicts = []
        for cur_section_name in self.timing_log:
            section_dict = {'section':cur_section_name}
            section_times = self.timing_log[cur_section_name]
            section_dict["start_time"] = section_times["start_time"]
            section_dict["finish_time"] = section_times["finish_time"]
            duration = section_times["finish_time"] - section_times["start_time"]
            section_dict["duration_min"] = duration.total_seconds() / 60.0
            section_dict["duration_hr"] = duration_min / 60.0
            section_dict["rate_min"] = self.num_contracts() / duration_min
            section_dict["rate_hr"] = self.num_contracts() / duration_hr
            section_dicts.append(section_dict)
        # And now make it a DataFrame, telling pandas that it's a list of rows,
        # and save to .csv
        plu.dict_list_to_csv(section_dicts, self.get_timing_log_fpath())

    def stream_art_data(self, verbose=False):
        """ Used by the LDA pipeline, to stream individual Articles rather than
        Contracts. The latter is a bit janky since you then have to go in and
        extract the article list from the contract object. """
        for cur_contract_data in self.stream_split_contracts(verbose=verbose):
            print(cur_contract_data["contract_id"])
            # Get the list of articles
            art_data_list = plu.articles_as_strlist(cur_contract_data, tuples=True)
            # And now yield each element of *this* list itself (hence difference
            # between this function and stream_split_contracts())
            for cur_art_data in art_data_list:
                yield cur_art_data

    def stream_cleaned_art_data(self):
        """ Basically a wrapper around stream_articles() that also runs preprocessing
        on the articles before returning them.

        Added 2019-03-07 by JJ """
        for cur_art_data in self.stream_art_data():
            # In these tuples, [0] is the text and [1] is the metadata
            cur_art_data = (self.preprocess_text(cur_art_data[0]), cur_art_data[1])
            # Uncomment to save the cleaned article so it can be used by
            # construct_corpus. BUT warning this creates a LOT of files...
            #article_filename = "article_" + str(article_num) + ".pkl"
            #article_fpath = os.path.join(self.get_preprocessed_path(),
            #                                article_filename)
            #plu.safe_to_pickle(cleaned_doc, article_fpath)
            yield cur_art_data

    def stream_parses(self):
        # Loads and yields the next *statements* .pkl file, returning a dictionary
        # with keys "contract_id", "statements", and "lang"
        parses_fpath = self.get_parses_fpath()
        self.iprint("Streaming parses from " + parses_fpath)
        if not os.path.isfile(parses_fpath):
            raise Exception("Something has gone wrong: parsed pickle file "
                            + parses_fpath + " doesn't exist")
        parses_list = joblib.load(parses_fpath)
        for cur_parse in parses_list:
            yield cur_parse

    def stream_pdata(self):
        # Yield the statement_pdata, one chunk file for each iteration
        pdata_path = self.get_pdata_path()
        fpath_data = plu.sort_by_suffix(pdata_path)
        print(fpath_data[0])
        for df_num, cur_fpath in fpath_data:
            cur_df = pd.read_pickle(cur_fpath)
            yield df_num, cur_df

    def stream_preprocessed_contracts(self):
        """ Like stream_preprocessed_articles() but a wrapper around stream_pt_contracts()
        not stream_articles

        Added 2019-03-07 by JJ """
        for contract_num, cur_contract in enumerate(self.stream_pt_contracts()):
            cleaned_doc = self.preprocess_contract(cur_contract)
            yield cleaned_doc

    def stream_pt_contract_data(self, start=0, end=None, pt_only=False):
        """
        Yields a stream of contracts created with only the plaintext path.
        In other words, this streams the "base" Contract objects that will
        then be filled over the remainder of the pipeline

        Set pt_only=True if you just want a string containing the plaintext of the contract.
        Otherwise (the default), you get a stream of Contract data dicts (keys=id,lang,text)
        """
        self.dprint("running stream_pt_contract_data()")
        if self.use_mongo:
            client = MongoClient()
            db = client["contracts"]
            # Make sure the collection exists
            coll_name = self.get_corpus_name()
            if coll_name not in db.list_collection_names():
                raise Exception(f"No MongoDB collection named {coll_name}")
            collection = db[self.get_corpus_name()]
            for cur_doc in collection.find():
                cur_id = cur_doc["_id"]
                cur_lang = cur_doc["lang"]
                cur_text = cur_doc["text"]
                if pt_only:
                    yield cur_text
                cur_contract_data = {'contract_id':cur_id, 'lang':cur_lang, 'text':cur_text}
                yield cur_contract_data
        else:
            # Stream from disk
            all_fpaths = self.get_plaintext_fpaths()
            #self.dprint("all_fpaths:")
            #self.dprint(all_fpaths)
            for cur_fpath in all_fpaths:
                cur_text = plu.load_txt_by_fpath(cur_fpath)
                if pt_only:
                    yield cur_text
                # Otherwise get all the info from the filename
                cur_id = plu.get_contract_id(cur_fpath, is_fpath=True)
                cur_lang = plu.get_contract_lang(cur_fpath, is_fpath=True)
                cur_contract_data = {'contract_id':cur_id, 'lang':cur_lang, 'text':cur_text}
                yield cur_contract_data

    def stream_split_contracts(self, verbose=False, ext="json"):
        # Loads+yields the next artsplit file (which contains the contract_id
        # and language along with the plaintext and artsplits)
        if self.batch_range:
            self.iprint("Using provided batch_range")
            start = self.batch_range[0]
            end = self.batch_range[1]
        else:
            self.iprint("Streaming full set of split contracts")
            start = 0
            # end = None means just go to the end of the list
            end = None
        pkl_fpaths = glob.glob(os.path.join(self.get_artsplit_path(),f"*.{ext}"))
        pkl_fpaths.sort()
        self.iprint(f"Loading {ext} files from {self.get_artsplit_path()}")
        if end is not None:
            path_sublist = [fpath for fnum, fpath in enumerate(pkl_fpaths) 
                            if (fnum >= start) and (fnum <= end)]
        else:
            path_sublist = [fpath for fnum, fpath in enumerate(pkl_fpaths) if fnum >= start]
        # (The actual loading + yielding)
        for json_fpath in path_sublist:
            if verbose:
                self.iprint("Loading " + str(json_fpath))
            # Load the pickle
            contract_data = plu.json_load(json_fpath)
            yield contract_data

    def stream_statement_auth(self):
        stauth_path = self.get_statement_auth_fpath()
        cname = self.get_corpus_name()
        fpaths = [(n,os.path.join(stauth_path,f"04_{cname}_auth_{n}.pkl")) for n in range(self.num_chunks)]
        for chunk_num, cur_fpath in fpaths:
            chunk_df = pd.read_pickle(cur_fpath)
            yield chunk_num, chunk_df

    def test_plaintext_fpaths(self):
        all_fpaths = self.get_plaintext_fpaths()
        match_list = [plu.has_langcode(cur_fpath) for cur_fpath in all_fpaths]
        if not all(match_list):
            # Some non-matching filenames :(
            self.iprint("Some filenames have no language code, so assuming default_lang "
                  + "(" + str(self.DEFAULT_LANG) + ")")

    def update_metadata(self, meta_df, varcode):
        """ meta_df is the Pandas dataframe containing the new metadata, while varcode
        is a short code describing what the latest merge was (for example, "elections") """
        new_fpath = self.get_new_meta_fpath(varcode)
        plu.safe_to_pickle(meta_df, new_fpath)
        # Also generate .csv and .dta versions
        # Go into the codebook and get the labels for the vars *currently* in the metadata
        labels = {k:self.codebook[k] for k in list(self.codebook.keys()) if k in meta_df.columns}
        plu.safe_to_stata(meta_df, new_fpath.replace(".pkl",".dta"), write_index=False,
                          variable_labels=labels)
        plu.safe_to_csv(meta_df, new_fpath.replace(".pkl",".csv"), index=False)
        self.iprint(f"Saved {varcode} metadata file to {new_fpath}")
        self.meta_updated = True
        self.latest_meta_fpath = new_fpath

######################################
####### (FAKE) GETTERS AND SETTERS ###
######################################
# (Fake because they actually *generate* the paths, whereas getters are supposed
# to just *return* instance variables)
    ### "Primary" (most important) variables
    def get_corpus_name(self):
        return self.corpus_name
    def get_corpus_path(self):
        return self.corpus_path
    def get_plaintext_path(self):
        return os.path.join(self.get_corpus_path(),"txt")
    def get_data_path(self):
        return self.data_path
    def get_output_path(self):
        return self.output_path
    ### Logs
    def get_log_fpath(self):
        return os.path.join("logs",self.get_corpus_name() + "_log.txt")
    def get_timing_log_fpath(self):
        return self.get_log_fpath().replace("_log.txt","_times.txt")
    ### 1. Main pipeline files
    def get_debug_path(self):
        return os.path.join(self.get_output_path(),"debug")
    def get_artsplit_path(self):
        return os.path.join(self.get_output_path(),"01_artsplit")
    def get_artsplit_html_path(self):
        """ Output directory (only used if artsplit_debug == True) """
        return os.path.join(self.get_debug_path(), "artsplit_html")
    #def get_coref_pkl_path(self):
    #    return os.path.join(self.get_pkl_path(),"02_coref_resolved")
    def get_parse_path(self):
        return os.path.join(self.get_output_path(),"02_parses")
    # (We don't need an 01_articles csv folder, the parses are the first
    # thing we care about making human-readable)
    def get_parsed_csv_fpath(self):
        return os.path.join(self.get_output_path(),"02_" + self.get_corpus_name()
                            + "_parsed.csv")
    def get_parses_fpath(self):
        return os.path.join(self.get_output_path(),f"02_{self.get_corpus_name()}_parsed.pkl")
    ## Authority measures, statement-level
    def get_pdata_path(self):
        return os.path.join(self.get_output_path(),"03_pdata")
    def get_pdata_fpath(self):
        return os.path.join(self.get_output_path(),f"03_{self.get_corpus_name()}_pdata.pkl")
    def get_statement_auth_path(self):
        return os.path.join(self.get_output_path(),f"04_auth")
    def save_pdata_df(self, pdata_df, chunk_num=None):
        suffix = ""
        if chunk_num is not None:
            suffix = "_" + str(chunk_num)
        pdata_fpath = os.path.join(self.get_pdata_path(), f"{self.get_corpus_name()}_pdata{suffix}.pkl")
        plu.safe_to_pickle(pdata_df, pdata_fpath)
    def save_statement_auth(self, st_auth_df, chunk_num=None):
        if chunk_num is not None:
            suffix = f"_{chunk_num}"
        else:
            suffix = ""
        st_auth_fname = f"04_{self.get_corpus_name()}_auth{suffix}.pkl"
        st_auth_fpath = os.path.join(self.get_statement_auth_path(),st_auth_fname)
        print(f"Saving statement-level auth measures to {st_auth_fname}")
        plu.safe_to_pickle(st_auth_df, st_auth_fpath)
    def get_authsums_fpath(self, extension="csv"):
        return os.path.join(self.get_output_path(),"05_" + self.get_corpus_name()
                            + "_authsums." + extension)
    def save_authsums_df(self, authsums_df):
        authsums_fpath = os.path.join(self.get_output_path(),
                                      f"05_{self.get_corpus_name()}_authsums.pkl")
        self.iprint(f"Saving {os.path.basename(authsums_fpath)}")
        authsums_df.to_pickle(authsums_fpath)
        authsums_df.to_csv(authsums_fpath.replace(".pkl",".csv"))
        authsums_df.to_stata(authsums_fpath.replace(".pkl",".dta"))
    ## Auth measures plus summary stats
    def get_sumstats_fpath(self, extension="csv"):
        return os.path.join(self.get_output_path(),"06_" + self.get_corpus_name()
                            + "_sumstats." + extension)
    ### 3. JSON files (for testing)
    def get_artsplit_json_fpath(self):
        """ The path to the file containing the manual annotations, one contract
        json string per line """
        return os.path.join(".","annotations",self.get_corpus_name()+"_annos.json")
    def get_artsplit_json_output_fpath(self):
        """ Saved to "auto_artsplits.json" so it can be easily matched up with
        "<corpus>_annos.json", the manually-annotated version of the corpus that
        lives in the ./annotations subfolder """
        return os.path.join(self.get_output_path(),
                            self.get_corpus_name()+"_annos_auto.json")
    ### 4. Plaintext files (for *debugging*)
    def get_secsplit_text_path(self):
        """ Path for the human-readable section-parsed contracts """
        return os.path.join(self.get_output_path(),"secsplit_plaintexts")
    def get_plaintext_output_path(self):
        """ Path where copies of the plaintexts should be outputted to, if
        debug_sections is turned on """
        return os.path.join(self.get_output_path(),"orig_plaintexts")

    ### Metadata pipeline paths
    def get_raw_meta_df(self):
        """ This returns the filepath of the basic metadata-only file """
        return pd.read_csv(os.path.join(self.get_data_path(), self.get_corpus_name() + "_meta.csv"))
    def get_latest_meta_df(self):
        """ Gets the fpath of the most recent file produced by the meta pipeline """
        if not self.meta_updated:
            self.iprint("Loading raw metadata")
            return self.get_raw_meta_df()
        else:
            self.iprint(f"Loading {self.latest_meta_fpath}")
            return pd.read_pickle(self.latest_meta_fpath)
    def get_new_meta_fpath(self, varcode):
        return os.path.join(self.get_data_path(), self.get_corpus_name() +
                            f"_meta_{varcode}.pkl")
    def get_inflation_df(self):
        return pd.read_csv(self.INFLATION_FPATH)
    def get_strike_df(self):
        return pd.read_excel(self.STRIKE_FPATH)
    def get_merged_meta_fpath(self, extension="csv"):
        """ This returns the filepath where the data containing auth measures merged
        with the metadata should be saved """
        return os.path.join(self.get_output_path(), self.get_corpus_name() +
                            "_auth_meta." + extension)
    def get_liwc_counts_fpath(self, extension="csv"):
        """ This is the filepath for the *non*-merged version of the LIWC counts.
        For the merged version, use get_merged_liwc_filepath(). It takes an
        additional extension argument, so we can easily obtain the filepaths for
        both the .pkl and .csv version """
        return os.path.join(self.get_data_path(), 
            self.get_corpus_name() + "_liwc_counts." + extension)
    def get_statement_subj_fpath(self, extension="csv"):
        return os.path.join(self.get_data_path(), 
            self.get_corpus_name() + "_statement_subj." + extension)

    ### LDA pipeline paths
    def get_lda_path(self):
        if self.lda_on_obj_branches:
            return os.path.join(self.get_output_path(),"obranch_lda_k" + str(self.num_topics))
        else:
            return os.path.join(self.get_output_path(),"lda")
    def get_obranch_fpath(self, extension="csv"):
        """ Filepath for the dataframe containing just unique IDs and then the
        (space-separated) object branches for each statement """
        return os.path.join(self.get_lda_path(),"01_" + self.get_corpus_name() 
                            + "_obranches." + str(extension))
    def get_preprocessed_fpath(self):
        """ Pickle file where the preprocessed docs should be stored (so that they
        can be quickly loaded by construct_corpus) """
        return os.path.join(self.get_lda_path(),"02a_" + self.get_corpus_name()
                            + "_preproc_articles.pkl")
    def get_preprocessed_df_fpath(self):
        """ If we're doing LDA on the object branches, this pickle file stores
        the df where each row is the preprocessed version of an "article" (aka
        a concatenation of object branches) """
        return os.path.join(self.get_lda_path(),"02b_" + self.get_corpus_name()
                            + "_preproc_df.pkl")
    def get_lda_dict_fpath(self):
        return os.path.join(self.get_lda_path(),"02c_" + self.get_corpus_name()
                            + "_ldadict.pkl")
    def get_lda_doclist_fpath(self, doclist_num):
        """ The filepath for the doclists, which are just DFs containing the
        tokenized versions of each doc in the (eventual) subcorpus """
        return os.path.join(self.get_lda_path(),"03_" + self.get_corpus_name()
                            + "_ldadoclist_" + str(doclist_num) + ".pkl")
    def get_lda_subcorp_fpath(self, subcorp_num):
        return os.path.join(self.get_lda_path(),"06_" + self.get_corpus_name() 
                            + "_ldasubcorp_" + str(subcorp_num) + ".pkl")
    def get_lda_corpus_fpath(self):
        return os.path.join(self.get_lda_path(),"07_" + self.get_corpus_name() 
                            + "_ldacorpus_full.pkl")
    def get_lda_model_fpath(self):
        return os.path.join(self.get_lda_path(),"08_" + self.get_corpus_name()
                            + "_ldamodel.pkl")
    def get_topic_list_fpath(self, extension="txt"):
        """ Where to save the (plaintext) list of the most relevant words for 
        each topic. i.e., the plaintext we look at to come up with labels for
        each topic. """
        return os.path.join(self.get_lda_path(), self.get_corpus_name() + "." + str(extension))
    def get_lda_weights_fpath(self, extension="pkl"):
        """ The filepath for the LDA weights for each "document" (article) """
        return os.path.join(self.get_lda_path(),"09_" + self.get_corpus_name()
                            + "_lda_weights." + str(extension))
    def get_snauth_statement_fpath(self, extension="pkl"):
        """ The statement-level unweighted subnorm-specific auth measures """
        return os.path.join(self.get_lda_path(),"10a_" + self.get_corpus_name()
                            + "_statement_auth." + str(extension))
    def get_subnorm_auth_fpath(self, extension="pkl"):
        """ The filepath for the article-level (unweighted) subnorm-specific
        authority measures """
        return os.path.join(self.get_lda_path(),"10b_" + self.get_corpus_name()
                            + "_snauth." + str(extension))
    def get_weighted_auth_fpath(self, extension="csv"):
        """ The filepath of the weighted authority measures """
        return os.path.join(self.get_lda_path(),"11_" + self.get_corpus_name()
                            + "_weighted_auth." + str(extension))
    def get_contract_auth_fpath(self, extension="csv"):
        """ The final weighted authority measures! """
        return os.path.join(self.get_lda_path(),"12_" + self.get_corpus_name()
                            + "_summed_auth." + str(extension))
    def get_snauth_weights_fpath(self, subject, extension="pkl"):
        """ A dictionary, mapping topics to weighted auth measures for each subject """
        return os.path.join(self.get_lda_path(),"13_" + self.get_corpus_name()
                            + "_" + str(subject)  + "_weights." + str(extension))

#######################
####### CONSTRUCTOR ###
#######################
    def __init__(self, corpus_name, corpus_path=None, data_path=None,
                 output_path=None, use_sqlite=True, use_mongo=True, num_chunks=20,
                 default_subnorms=False, ocr=OCR_OPTIONS[0], parser=PARSER_OPTIONS[0],
                 num_topics=20, lang_list=["eng"], corp_extensions=["txt"],
                 debug=False, in_memory=False, default_encoding="utf-8", id_list_fpath=None,
                 sample_N=None, sample_range=None, random_seed=1948, sample_meta=True,
                 use_logging=False, use_aws=False, pause_for_input=False,
                 progress_bars=True, debug_artsplit=False, eval_artsplit=False,
                 batch_range=None, lda_on_obj_branches=True, num_lda_subsets=1,
                 num_lda_workers=36, lda_subcorp_num=0, force_overwrite=False,
                 break_id=None, recurse_corpus_dir=True, split_method="regex"):
        """
        The class constructor. The only required argument is corpus name.
        Everything else is an optional argument changing settings/configuration.
        [Technically this means the corpus_path and data_path are required, but
        they don't have to be given in the constructor, since they can be loaded
        from the .conf file]
        """
        # First things first, make sure our working directory is the directory
        # that contains this (pipeline.py) file
        os.chdir(self.PIPELINE_PATH)
        # The most important pipeline parameters
        self.corpus_name = corpus_name
        # The db name is just the corpus name (so, canadian will be canadian.sqlite)
        self.db_name = corpus_name
        # Which OCR engine to use
        self.ocr = ocr
        # Which parser to use
        self.parser = parser
        # The number of topics to use for the LDA pipeline
        self.num_topics = num_topics
        # The number of chunks to split the big datasets into, to avoid running out
        # of memory
        self.num_chunks = num_chunks
        # If True, pauses between each step of the pipeline so you can sanity-check the output
        self.pause_for_input = pause_for_input
        # self.use_aws should be True if we're running parts of the pipeline on AWS,
        # False otherwise (if it's all being run on textlab)
        self.use_aws = use_aws
        # [NOTE: setting both debug and progress_bars to True will have weird
        # effects, since the tqdm progress bar library depends on stdout
        # *not* getting anything printed to it...]
        self.debug = debug
        # User can set a specific contract id, and have the pipeline steps break
        # on that id
        self.break_id = break_id
        self.progress_bars = progress_bars
        # See also: debug artsplit
        self.eval_artsplit = eval_artsplit
        # Make sure everything in batch_range is an int
        if batch_range and batch_range[0]:
            if not type(batch_range[0]) == int:
                raise ValueError("Elements of batch_range must be of type int or None")
            if not type(batch_range[1]) == int:
                raise ValueError("Elements of batch_range must be of type int or None")
        self.batch_range = batch_range
        # If True, the pipeline doesn't ask before overwriting important files
        self.force_overwrite = force_overwrite
        # If True, LDA is run only on object branches. Otherwise, it is run on
        # full sections.
        self.lda_on_obj_branches = lda_on_obj_branches
        # The number of "chunks" to cut the LDA corpus into, which should
        # correspond to the number of different servers you'll be running the
        # pipeline on
        self.num_lda_subsets = num_lda_subsets
        # The number of cores the instance should use to estimate the LDA model
        self.num_lda_workers = num_lda_workers
        # The subcorpus that this AWS instance should work on (Should be None if
        # just a normal TextLab run)
        self.lda_subcorp_num = lda_subcorp_num
        # How to do the section splits (simple RegEx or Elliott's parser)
        self.split_method = split_method
        # If True, artsplit results are outputted to the data/debug folder
        self.debug_artsplit = debug_artsplit

        # All of the paths used by the pipeline are derived from these two
        # *static* locations (corpus_path and data_path), which should be loaded
        # from the .conf file for "regular" runs but can be supplied to the
        # constructor for quick test runs. 
        # If the paths aren't given to the constructor and if no .conf file exists,
        # throws an Exception (in load_config_paths())
        path_data = plu.load_config_pathdata(self.corpus_name)
        self.corpus_path, self.data_path, self.output_path = path_data
        # Now let the constructor args overwrite the .conf file
        if corpus_path is not None:
            self.corpus_path = corpus_path
        # Quick check: there needs to be a ./txt subdir within corpus_path dir
        if not os.path.isdir(os.path.join(self.corpus_path,"txt")):
            raise Exception("Plaintexts must be stored in a /txt subdirectory of " +
                            "the given corpus directory (" + self.corpus_path + ")")
        if data_path is not None:
            self.data_path = data_path
        if output_path is not None:
            self.output_path = output_path

        self.use_logging = use_logging
        if self.use_logging:
            # Set logging level based on debug. The DEBUG level will mostly
            # be used for things like timing, while INFO will be used to
            # "track" where you are in the pipeline. See:
            # https://stackoverflow.com/questions/13733552/
            # logger-configuration-to-log-to-file-and-print-to-stdout
            if debug:
                self.logging_level = logging.DEBUG
            else:
                self.logging_level = logging.INFO
            log_formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(message)s')
            root_logger = logging.getLogger()
            root_logger.setLevel(self.logging_level)
            # The old code (just this one line)
            #logging.basicConfig(format=log_formatter, level=self.logging_level)
            # And set it up to *also* export the log messages to a file
            if not os.path.isdir("logs"):
                os.mkdir("logs")
            log_fpath = os.path.join("logs",self.corpus_name + ".log")
            log_filehandler = logging.FileHandler(log_fpath)
            log_filehandler.setFormatter(log_formatter)
            root_logger.addHandler(log_filehandler)
            # And then the regular console logger
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_formatter)
            root_logger.addHandler(console_handler)

        # If default_subnorms is False, go back to .conf file to load corpus-specific
        # subjectnorms (as an OrderedDict)
        if not default_subnorms:
            self.subnorm_map = plu.load_config_subnorms(self.corpus_name)
        else:
            self.subnorm_map = plu.DEFAULT_SUBNORM_MAP
        # And derive the (non-normalized) subject list
        self.subject_list = list(self.subnorm_map.keys())
        # And the subnorm list, which is a bit trickier
        self.subnorm_list = list(self.subnorm_map.values()) + ["other"]
        # Remove the duplicates, retaining order (kinda a weird trick)
        self.subnorm_list = list(OrderedDict.fromkeys(self.subnorm_list))

        # lang_list: see the GitHub Wiki for the list of 
        # 2-letter language codes
        self.lang_list = lang_list

        # This is a down-the-road TODO: if True, just work in memory rather
        # than exporting (.pkl/.csv) files after each pipeline step. This
        # should save a bunch of time (we spend a lot of time exporting and
        # importing data)
        self.in_memory = in_memory

        # This encoding param will become important if we start to parse
        # (for example) Chinese contracts
        self.default_encoding = default_encoding

        # This param is important if we get a corpus with weird file extensions
        # (or for example if we want to support .html files intead of just .txt)
        self.corp_extensions = corp_extensions

        # (Mongo for documents only -- the rest [should] use SQLite3)
        self.use_mongo = use_mongo

        # If True (default), pipeline searches for .txt plaintext filse in the
        # corpus directory plus all subdirectories. Otherwise, it just searches
        # the corpus directory and *ignores* subdirectories.
        self.recurse_corpus_dir = recurse_corpus_dir

        ### Pipelines

        # (Ordered) dictionary of the run() functions for all possible pipelines
        self.pipeline_fns = OrderedDict()
        self.pipeline_fns["main"] = self.run_main
        self.pipeline_fns["meta"] = self.run_meta
        self.pipeline_fns["lda"] = self.run_lda

        # A variable containing the most recently exported .csv file of merged
        # metadata. This is defined so that you can skip steps in the metadata
        # merge pipeline if you only want specific metadata (for example, so that
        # the labor_rates merge code doesn't just look "one step back" at the
        # liwc merge file if you skipped the LIWC merge step.)
        self.latest_meta_fpath = None
        # True iff the metadata was updated, i.e., merged with additional datasets
        self.meta_updated = False
        # Load codebook.csv and convert it into a dict
        self.codebook = self.load_codebook()

        # Make sure we have a var that can hold the contract_num to contract_id
        # crosswalk if that's needed
        self.id_df = None

        ### Sampling stuff. Needs to go at the end since all the other constructor
        ### vars need to be set first (so they can be modified here)
        self.sample_N = sample_N
        self.sample_range = sample_range
        self.id_list_fpath = id_list_fpath
        self.sample_meta = sample_meta
        self.random_seed = random_seed
        self.iprint("Seeding RNG: " + str(self.random_seed))
        random.seed(self.random_seed)
        # And now import and call sample_corpus() if need be
        if self.sample_N or self.sample_range or self.id_list_fpath:
            from create_sample_corpus import create_sample_corpus
            # create_sample_corpus returns a dict containing the new sample corpus
            # name, path, data_path, and contract_ids
            sample_data = create_sample_corpus(self)
            # So set these as our *pipeline*'s name, path, and data_path
            self.corpus_name = sample_data["corpus_name"]
            if "corpus_path" in sample_data:
                # (This is false if use_mongo is True, since in that case we don't
                # need to construct a new path on the filesystem)
                self.corpus_path = sample_data["corpus_path"]
            self.data_path = sample_data["data_path"]
            self.output_path = sample_data["output_path"]
            self.contract_ids = sample_data["contract_ids"]
        else:
            # If not sampling, set contract_ids to be the full set of
            # contract_ids
            self.contract_ids = self.get_contract_ids()

        # This is a "timing log" that will keep track of how long
        # each step takes. Then the compute_pace() function above (TODO) will
        # compute contract/hour rates and estimate how long the *full* corpus
        # would take (if the current run is on a sample)
        self.log_time("init")

############################
####### PIPELINE RUNNERS ###
############################

    def run(self, steps=None):
        # Runs the *full* pipeline, including the "auxiliary" pipelines like
        # lda and metadata merging. But that means really it just calls lots
        # of other functions
        self.iprint("Starting full pipeline on corpus " + self.corpus_name)
        if steps:
            self.iprint("--> Steps: " + str(steps))
        if (steps is None) or (steps == "all"):
            steps = list(self.pipeline_fns.keys())
        for cur_pipeline_name in steps:
            # TODO: get substeps out of steps *if* its a dict, otherwise "all"
            # Check if steps is a *list* or a *dict*
            if isinstance(steps, list):
                substeps = "all"
            elif isinstance(steps, dict):
                substeps = steps[cur_pipeline_name]
            else:
                raise Exception("The 'steps' argument needs to be either a "
                                + "list or a dict")
            self.pipeline_fns[cur_pipeline_name](steps=substeps)
        self.iprint("Full run is complete!")

    def run_main(self, steps=None):
        # Runs the main pipeline (the .py files starting with "main")
        self.iprint(f"Running main pipeline on corpus {self.corpus_name}")
        cwd = os.getcwd()
        self.dprint(f"Working directory: {cwd}")
        # Function sequence for the *main* pipeline
        # Check if they specified a custom list of steps to run
        if steps:
            self.iprint("--> Steps: " + str(steps))
        else:
            # Run all steps if list not given explicitly
            self.iprint("--> Running all steps")
        def step_data(step_info):
            if isinstance(step_info, str):
                return (step_info, {})
            elif isinstance(step_info, tuple):
                # It's a tuple, with 0->name and 1->kwargs
                return step_info
            else:
                raise AttributeError("Step list must contain only strings or tuples")
        def get_mod_name(step_name):
            """ Given a step name, find the file containing that step """
            fname_glob = f"main*{step_name}.py"
            full_glob = os.path.join(self.PIPELINE_PATH, fname_glob)
            #print(f"full_glob = {full_glob}")
            mod_name = os.path.basename(glob.glob(full_glob)[0]).replace(".py","")
            return mod_name
        for cur_step in steps:
            cur_step_name, cur_step_kwargs = step_data(cur_step)
            # Now we need to import it
            # So first we need to find the file (module) that contains the step.
            # [Note: make sure we're looking in the same directory as *this* file,
            # i.e., PIPELINE_PATH]
            # Next we need to import it using importlib
            step_module = importlib.import_module(get_mod_name(cur_step_name))
            # Finally, pull out the specific function we want to call from step_module
            mod_fn = getattr(step_module, cur_step_name)
            # Now we call it with the specific kwargs
            mod_fn(self, **cur_step_kwargs)
            # And pause for input if needed
            if self.pause_for_input:
                input("Press Enter to Continue...")
        self.iprint("Main pipeline complete")

    def run_err_rates(self):
        # Runs the pipeline generating a .csv file with (sentence-level)
        # error rate stats for each subnorm
        self.iprint("Starting error rates pipeline")
        from sent_err_rates import sent_err_rates
        sent_err_rates(self)
        self.iprint("Error rates pipeline complete")

    def run_lda(self, steps=None):
        # Runs the LDA pipeline (the .py files starting with "lda")
        self.iprint("Starting LDA pipeline")
        # Add the "lda_pipeline" folder to path (temporarily) then import the
        # LDA functions from it
        lda_dir = "lda_pipeline"
        lda_path = os.path.join(os.path.dirname(__file__), lda_dir)
        sys.path.append(lda_path)
        if self.lda_on_obj_branches:
            from lda01_extract_object_branches import extract_object_branches
        from lda02_construct_dictionary import construct_dictionary
        from lda03_split_docs import split_docs
        if self.num_lda_subsets > 1:
            from lda04_export_to_aws import export_to_aws
            from lda05_remote_control_aws import remote_control_aws
        from lda06_construct_corpus import construct_corpus
        from lda07_combine_corpora import combine_corpora
        from lda08_run_lda import run_lda
        from lda09_compute_weights import compute_weights
        from lda10_construct_subnorm_auths import construct_subnorm_auths
        from lda11_weight_auth_measures import weight_auth_measures
        from lda12_sum_weighted_auths import sum_weighted_auths
        from lda13_generate_auth_graphs import generate_auth_graphs

        self.lda_fns = OrderedDict()
        if self.lda_on_obj_branches:
            self.lda_fns["extract_object_branches"] = extract_object_branches
        else:
            self.iprint("Skipping extract_object_branches since LDA is being run "
                        + "on full articles")
        self.lda_fns["construct_dictionary"] = construct_dictionary
        self.lda_fns["split_docs"] = split_docs
        if self.num_lda_subsets > 1:
            self.lda_fns["export_to_aws"] = export_to_aws
            self.lda_fns["remote_control_aws"] = remote_control_aws
        self.lda_fns["construct_corpus"] = construct_corpus
        self.lda_fns["combine_corpora"] = combine_corpora
        self.lda_fns["run_lda"] = run_lda
        self.lda_fns["compute_weights"] = compute_weights
        self.lda_fns["construct_subnorm_auths"] = construct_subnorm_auths
        self.lda_fns["weight_auth_measures"] = weight_auth_measures
        self.lda_fns["sum_weighted_auths"] = sum_weighted_auths
        self.lda_fns["generate_auth_graphs"] = generate_auth_graphs
        if steps:
            self.iprint("--> Steps: " + str(steps))
            if steps == "all":
                steps = self.lda_fns.keys()
        else:
            steps = self.lda_fns.keys()

        for cur_step in steps:
            # If it's just a string, run it
            if isinstance(cur_step,str):
                self.iprint("Calling " + str(cur_step))
                self.lda_fns[cur_step](self)
                self.iprint("Completed " + str(cur_step))
            else:
                # Get the custom arg
                cur_step_name = cur_step[0]
                arg_dict = cur_step[1]
                self.lda_fns[cur_step_name](self, **arg_dict)

            if self.pause_for_input:
                input("Press Enter to Continue...")

        self.iprint("LDA pipeline complete")

    def run_meta(self, steps=None):
        # Runs the metadata merge pipeline (.py files starting with "meta")
        self.iprint("Starting metadata merge pipeline")
        # Add the "meta_pipeline" folder to path (temporarily) then import the
        # meta fns from it
        meta_dir = "meta_pipeline"
        meta_path = os.path.join(os.path.dirname(__file__), meta_dir)
        sys.path.append(meta_path)
        from meta01_basic_vars import basic_vars
        from meta02_merge_inflation import merge_inflation
        from meta03_merge_election_data import merge_election_data
        from meta04_merge_labor_rates import merge_labor_rates
        from meta05_link_contract_series import link_contract_series
        from meta06_merge_strikes import merge_strikes
        #from meta05_merge_wms_scores import merge_wms_scores
        #from meta06_plot_sn_props import plot_sn_props

        # Function sequence
        self.meta_fns = OrderedDict()
        self.meta_fns["basic_vars"] = basic_vars
        self.meta_fns["merge_inflation"] = merge_inflation
        self.meta_fns["merge_election_data"] = merge_election_data
        self.meta_fns["merge_labor_rates"] = merge_labor_rates
        self.meta_fns["link_contract_series"] = link_contract_series
        self.meta_fns["merge_strikes"] = merge_strikes
        #self.meta_fns["merge_wms_scores"] = merge_wms_scores
        #self.meta_fns["plot_sn_props"] = plot_sn_props

        # Run the desired sequence steps
        if steps:
            self.iprint("--> Steps: " + str(steps))
            if steps == "all":
                steps = self.meta_fns.keys()
        else:
            steps = self.meta_fns.keys()
        for cur_step_name in steps:
            self.iprint("Calling " + cur_step_name)
            # Check if its a dict, where the value is itself a dict of keyword args
            if type(steps) == dict:
                # TODO: Make it work for general keyword args, not just input_fpath
                kw_args = steps[cur_step_name]
                kw_value = kw_args["input_fpath"]
                self.meta_fns[cur_step_name](self, input_fpath=kw_value)
            else:
                # Run the step without any kw args
                self.meta_fns[cur_step_name](self)
            self.iprint("Completed " + cur_step_name)
            # Pause if needed
            if self.pause_for_input:
                input("Press Enter to Continue...")

        self.iprint("Metadata pipeline complete")

###################################
### MAIN FUNCTION (FOR TESTING) ###
###################################

if __name__ =="__main__":
    # THIS IS ONLY FOR *TESTING* THE Pipeline CLASS -- A SEPARATE FILE
    # SHOULD BE CREATED FOR ACTUAL RUNS
    test = Pipeline("test_corpus")
    test.run_main()
    print(test.corpus_dir)
    print(test.data_dir)
