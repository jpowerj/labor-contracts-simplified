# Python imports
import glob
import os

# Local imports
import plutil

# 3rd party imports
import boto3
import numpy as np
import spacy

class Pipeline:
    def __init__(self, corpus_name, mode="plaintext", mode_options=None,
                 output_dirname=None, output_path=None, lang_list=["eng"], sample_N=None,
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
        :param output_dirname: str, optional
            The subdirectory, *within* the Labor_Contracts_Canadian output
            directory, where the output files will be saved.
            By default, a directory is generated based on the timestamp of when
            the constructor is run.
        :param output_path: str, optional
            This parameter, unlike `output_dirname`, overrides the default output
            directory and instead saves everything into `output_path`
        :param lang_list: `list` of str, optional
        :param sample_N: int, optional
        :param random_seed: int, optional
        :param splitter: str, optional
        :param verbose: bool, optional
        """
        self.verbose = verbose
        self.vprint = print if self.verbose else lambda x: None
        self.corpus_name = corpus_name
        self.output_dirname = plutil.gen_timestamp_str()
        if output_dirname:
            self.output_dirname = output_dirname
        self.output_path = None
        if output_path:
            self.output_path = output_path
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
        # We don't initialize the spaCy model here (only as needed)
        self.spacy_model = None

    def get_output_path(self):
        """
        Construct the output path based on the corpus name, assuming that we're
        in Labor_Contracts_Canada/analysis/code/code-python/simplified_pipeline/.

        :return: str
            The relative path to the output directory
        """
        if self.output_path:
            return os.path.join(self.output_path, self.output_dirname)
        else:
            # We just use dirname
            return os.path.join("..", "..", "..", "output", "analysis", self.output_dirname)

    def get_artsplit_output_path(self, ext):
        return os.path.join(self.get_output_path(), f"01_artsplit_elliott_{ext}")

    def get_spacy_output_path(self):
        return os.path.join(self.get_output_path(), "02_spacy_pkl")

    def get_sdata_output_path(self):
        return os.path.join(self.get_output_path(), "03a_sdata_pkl")

    def get_pdata_output_path(self):
        return os.path.join(self.get_output_path(), "03b_pdata_pkl")

    def get_num_docs(self):
        return len(self.txt_fnames)

    def _get_pt_fnames_lang(self, lang):
        """ Get plaintext fnames for a specific language """
        pt_glob_str = os.path.join(self.plaintext_path, f"*_{lang}.txt")
        lang_fpaths = sorted(glob.glob(pt_glob_str))
        lang_fnames = [os.path.basename(fpath) for fpath in lang_fpaths]
        return lang_fnames

    def get_spacy_model(self):
        """
        Annoying but necessary additional step: adding "contract_id" and
        "art_num" attributes to spacy's Doc class, so that we can serialize and
        deserialize without headaches

        See https://spacy.io/usage/processing-pipelines#custom-components-attributes

        All the neuralcoref attributes for a doc, for future reference:
        * `cr_test_doc._.has_coref`
        * `cr_test_doc._.coref_resolved`
        * `cr_test_doc._.coref_clusters`
        * `cr_test_doc._.coref_scores`

        And code for looping over the clusters:
        ```
        for cluster in cr_test_doc._.coref_clusters:
            print(f"===== #{cluster.i}")
            print(cluster)
            print(f"main: '{cluster.main}'")
            print(cluster.mentions)
            for mention in cluster.mentions:
                print(mention)
                print(mention.start)
                print(mention.end)

        :return: :obj:
            The spaCy model with custom `contract_id`, `lang`, and `art_num` fields
        """
        if self.spacy_model is None:
            self.vprint("Loading spaCy core model")
            nlp_eng = spacy.load('en_core_web_md', disable=["ner"])
            # The force=True is just so that we can change (e.g.) the names or
            # default values and overwrite the extensions (otherwise this would
            # always cause an Exception)
            #spacy.tokens.Doc.set_extension("contract_id", default=None, force=True)
            #spacy.tokens.Doc.set_extension("lang", default=None, force=True)
            #spacy.tokens.Doc.set_extension("art_num", default=None, force=True)
            self.spacy_model = nlp_eng
        return self.spacy_model

    def split_contracts(self):
        # Import the contract splitting code
        import main01_split_contracts
        # Do the splitting
        main01_split_contracts.split_contracts(self)

    def parse_articles(self):
        import main02_parse_articles
        main02_parse_articles.parse_articles(self)