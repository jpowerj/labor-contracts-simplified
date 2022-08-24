# Python imports
import glob
import os

# 3rd party imports
import numpy as np

class Pipeline:
    def get_output_path(self):
        # Construct the output path based on the corpus name
        output_path = os.path.join("..", self.corpus_name + "_output")
        return output_path
    
    def get_plaintext_fpaths(self):
        """ Get plaintext fpaths for all contracts with languages in lang_list """
        pass
    
    def _get_pt_fpaths_lang(self, lang):
        """ Get plaintext fpaths for a specific language """
        pt_glob_str = os.path.join(self.plaintext_path, f"*_{lang}.txt")
        lang_fpaths = sorted(glob.glob(pt_glob_str))
        return lang_fpaths
    
    def split_contracts(self):
        # Import the contract splitting code
        import main01_split_contracts
        # Do the splitting
        main01_split_contracts.split_contracts(self)
        
    def parse_articles(self):
        import main02_parse_articles
        main02_parse_articles.parse_articles(self)
        
    def __init__(self, corpus_name, plaintext_path, lang_list=["eng"], sample_N=None,
                 random_seed=1948, splitter="regex"):
        self.corpus_name = corpus_name
        self.lang_list = lang_list
        self.plaintext_path = plaintext_path
        # Need to get all the plaintext fpaths here, since (a) we need lang_list
        # for language filtering and (b) we may need to sample from it later
        # (based on the value of sample_N)
        self.plaintext_fpaths = []
        for cur_lang in self.lang_list:
            lang_fpaths = self._get_pt_fpaths_lang(cur_lang)
            self.plaintext_fpaths.extend(lang_fpaths)
        self.sample_N = sample_N
        self.random_seed = random_seed
        self.splitter = splitter
        if self.sample_N is not None:
            # Sample N contracts
            np.random.seed(self.random_seed)
            all_fpaths = self.plaintext_fpaths
            print(f"Sampling {sample_N} contracts")
            sample_fpaths = np.random.choice(all_fpaths, self.sample_N, replace=False)
            # Now we can replace plaintext_fpaths
            self.plaintext_fpaths = sample_fpaths
            # And update the corpus name
            self.corpus_name = self.corpus_name + "_s" + str(self.sample_N)