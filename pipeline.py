# Python imports
import glob
import os

# Local imports
import plutil

# 3rd party imports
import numpy as np
import boto3


class Pipeline:
    def __init__(self, corpus_name, mode="plaintext", mode_options=None,
                 output_dirname=None, lang_list=["eng"], sample_N=None,
                 random_seed=1948, splitter="regex", verbose=False):
        """
        Object containing data needed throughout the pipeline.

        :param corpus_name: str
            The name of the corpus, used for keeping track of data+output files
        :param mode: str
            's3' or 'plaintext'.
        :param mode_options:
            If `mode` == 's3', this specifies 'bucket' and 'prefix' keys.
            If `mode` == 'plaintext', this specifies 'plaintext_path'
        :param output_dir: str, optional
            The subdirectory, *within* the Labor_Contracts_Canadian output
            directory, where the output files will be saved.
            By default, a directory is generated based on the timestamp of when
            the constructor is run.
        :param lang_list: `list` of str, optional
        :param sample_N: int, optional
        :param random_seed: int, optional
        :param splitter: str, optional
        :param verbose: bool, optional
        """
        self.verbose = verbose
        self.vprint = print if self.verbose else lambda x: None
        self.corpus_name = corpus_name
        if output_dirname:
            self.output_dirname = output_dirname
        else:
            self.output_dirname = plutil.gen_timestamp_str()
        self.vprint(f"Full output path: {self.get_output_path()}")
        self.lang_list = lang_list
        if mode == "plaintext":
            self.plaintext_path = mode_options['plaintext_path']
            self.txt_fnames = []
            for cur_lang in self.lang_list:
                lang_fnames = self._get_pt_fnames_lang(cur_lang)
            self.txt_fnames.extend(lang_fnames)
        elif mode == "s3":
            self.s3_bucket_name = mode_options['bucket_name']
            self.s3_bucket_prefix = mode_options['bucket_prefix']
            self.vprint(f"Loading filenames from {self.s3_bucket_name}/{self.s3_bucket_prefix}...")
            self.txt_fnames, self.excluded_fnames = plutil.get_s3_contents(self.s3_bucket_name, self.s3_bucket_prefix,
                                                      lang_list=self.lang_list, return_excluded=True, verbose=self.verbose)
            self.vprint(f"{len(self.txt_fnames)} filenames loaded ({self.txt_fnames[0]} ... {self.txt_fnames[-1]})")
        # Need to get all the plaintext fpaths here, since (a) we need lang_list
        # for language filtering and (b) we may need to sample from it later
        # (based on the value of sample_N)
        self.sample_N = sample_N
        self.random_seed = random_seed
        self.splitter = splitter
        if self.sample_N is not None:
            # Sample N contracts
            np.random.seed(self.random_seed)
            self.vprint(f"Sampling {sample_N} contracts")
            sample_fnames = np.random.choice(self.txt_fnames, self.sample_N, replace=False)
            # Now we can replace plaintext_fpaths
            self.txt_fnames = sample_fnames
            # And update the corpus name
            self.corpus_name = self.corpus_name + "_s" + str(self.sample_N)

    def get_output_path(self):
        """
        Construct the output path based on the corpus name, assuming that we're
        in Labor_Contracts_Canada/analysis/code/code-python/simplified_pipeline/.

        :return: str
            The relative path to the output directory
        """
        return os.path.join("..", "..", "..", "output", "analysis", self.output_dir)

    def get_num_docs(self):
        return len(self.txt_fnames)

    def _get_pt_fnames_lang(self, lang):
        """ Get plaintext fnames for a specific language """
        pt_glob_str = os.path.join(self.plaintext_path, f"*_{lang}.txt")
        lang_fpaths = sorted(glob.glob(pt_glob_str))
        lang_fnames = [os.path.basename(fpath) for fpath in lang_fpaths]
        return lang_fnames

    def split_contracts(self):
        # Import the contract splitting code
        import main01_split_contracts
        # Do the splitting
        main01_split_contracts.split_contracts(self)

    def parse_articles(self):
        import main02_parse_articles
        main02_parse_articles.parse_articles(self)