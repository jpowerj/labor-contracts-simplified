# Python imports
import functools
import glob
import json
import logging
import os

# 3rd party imports
import joblib
import multiprocessing_logging
import spacy
import neuralcoref

# Local imports
import pipeline
import main02_parse_articles

def get_coref_data(doc_obj):
    mentions = [
        {
            "start": mention.start_char,
            "end": mention.end_char,
            "text": mention.text,
            "resolved": cluster.main.text,
        }
        for cluster in doc_obj._.coref_clusters
        for mention in cluster.mentions
    ]
    return mentions

def remove_unserializable_results(doc):
    doc.user_data = {}
    for x in dir(doc._):
        if x in ['get', 'set', 'has']: continue
        setattr(doc._, x, None)
    for token in doc:
        for x in dir(token._):
            if x in ['get', 'set', 'has']: continue
            setattr(token._, x, None)
    return doc

def stream_art_data(test_N=None):
    """
    test_N: If set to an int, the function will only yield article data for the first `test_N` contracts.
            Otherwise, if set to None, article data for all contracts is yielded.
    """
    art_data_fpaths = glob.glob("../canadian_output/01_artsplit_elliott_json/*.json")
    # Loop over contracts
    for fnum, fpath in enumerate(art_data_fpaths):
        if test_N is not None and fnum >= test_N:
            # We've already yielded the first `test_N` contracts, so terminate
            break
        with open(fpath, 'r') as f:
            all_articles = json.load(f)
        # Now loop over the articles
        for cur_article in all_articles:
            # We want to yield tuples of (string, {contract_id, article_num})
            art_str = cur_article['text']
            art_data = {'contract_id':cur_article['contract_id'],
                        'article_num':cur_article['section_num']}
            yield (art_str, art_data)
            
def transform_texts(nlp, batch_id, batch_tuples, output_dir):
    # Using spacy's "DocBin" functionality: see https://spacy.io/usage/saving-loading#docs
    batch_bin = spacy.tokens.DocBin(store_user_data=True)
    #print(nlp.pipe_names)
    output_fpath = os.path.join(output_dir, f"{batch_id}.bin")
    if os.path.isfile(output_fpath):  # return None in case same batch is called again
        return None
    print("Processing batch", batch_id)
    for art_doc, art_meta in nlp.pipe(batch_tuples, as_tuples=True):
        # This is the weird part where we now have to change contract_id and art_num
        # from being metadata to being attributes of the spacy Doc objects themselves
        contract_id = art_meta["contract_id"]
        article_num = art_meta["article_num"]
        art_doc._.contract_id = contract_id
        art_doc._.article_num = article_num
        # And now we don't need the meta object anymore, since it's encoded in the Doc itself
        # But next we need to get a serializable representation of the detected corefs
        art_doc._.coref_list = get_coref_data(art_doc)
        # Ok now we can get rid of the original coref attributes that break the data
        art_doc = remove_unserializable_results(art_doc)
        batch_bin.add(art_doc)
    # Now we can use spacy's serialization methods [joblib basically fails at serializing
    # spacy Docs for various reasons]
    # [see https://spacy.io/usage/saving-loading#docs]
    batch_bytes = batch_bin.to_bytes()
    # And save the bytes object to file
    with open(output_fpath, "wb") as f:
        f.write(batch_bytes)
    print("Saved {} texts to {}.bin".format(len(batch_tuples), batch_id))

def main():
    # Set up logging
    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    # And set it to work with spacy's use of multiprocessing
    multiprocessing_logging.install_mp_handler()
    print("Loading spaCy core model")
    nlp_eng = spacy.load('en_core_web_md', disable=["ner"])
    print("Loading spaCy coref model. May take a while...")
    neuralcoref.add_to_pipe(nlp_eng)
    # The force=True is just so that we can change (e.g.) the names or default values and overwrite the extensions
    # (otherwise this would always cause an Exception)
    spacy.tokens.Doc.set_extension("contract_id", default=None, force=True)
    spacy.tokens.Doc.set_extension("article_num", default=None, force=True)
    spacy.tokens.Doc.set_extension("coref_list", default=[], force=True)
    
    # Trying to use multiprocessing like in
    # https://spacy.io/usage/examples#multi-processing
    output_dir = "./mp_full"
    art_tuple_stream = stream_art_data()
    print("Processing texts...")
    batch_size = 1000
    #batch_size = 200
    n_jobs = 16
    art_partitions = spacy.util.minibatch(art_tuple_stream, size=batch_size)
    executor = joblib.Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    do = joblib.delayed(functools.partial(transform_texts, nlp_eng))
    tasks = (do(i, batch_tuples, output_dir) for i, batch_tuples in enumerate(art_partitions))
    executor(tasks)
    
if __name__ == "__main__":
    main()