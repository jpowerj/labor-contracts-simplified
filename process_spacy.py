from collections import defaultdict

from timeit import default_timer as timer

subject_deps = [
    'nsubj',
    'nsubjpass',
    'expl'
]

main_deps = subject_deps + [
    'advmod',
    'dobj',
    'prep',
    'xcomp',
    'dative', # indirect object
    'advcl',
    'agent',
    'ccomp',
    'acomp',
    'attr'
]

def get_branch(t, sent, include_self=True):
    branch = recurse_tokens(t)
    if include_self:
        branch += [t]
    #branch = [m for m in branch if m.dep_ != 'punct' and not m.orth_.isdigit()]
    branch = [w for w in sent if w in branch] # and w.dep_ in include]
    lemmas = []
    tags = []
    for token in branch:
        lemma = token.lemma_.lower()
        #if len(lemma) <= 2:
        #    continue
        if any([char.isdigit() for char in lemma]):
            continue
        if any(punc in lemma for punc in ['.',',',':',';', '-']):
            continue
        lemmas.append(lemma)
        tags.append(token.tag_)
    #tags = [w.tag_ for w in sent if w in mods]
    return lemmas, tags


def get_statements(art_doc, contract_id, art_num, resolve_corefs=False):
    """

    :param art_nlp:
    :param contract_id:
    :param art_num:
    :return:
    """
    #print("get_statements()")
    statement_list = []
    time_in_pbs = 0
    if resolve_corefs:
        # For now, since spaCy neural coref is super buggy, need to check if
        # there are any coref clusters in the doc
        any_corefs = art_doc._.coref_clusters is not None
        if not any_corefs:
            resolve_corefs = False
    # Loop over sentences in the article
    for sentence_num, sent in enumerate(art_doc.sents):
        tokcheck = str(sent).split()
        if any([x.isupper() and len(x) > 3 for x in tokcheck]):
            # Don't parse this sentence
            continue

        pbs_start = timer()
        sent_statements = parse_by_subject(sent, resolve_corefs=resolve_corefs)
        pbs_end = timer()
        pbs_elapsed = pbs_end - pbs_start
        time_in_pbs += pbs_elapsed

        for statement_num, statement_data in enumerate(sent_statements):
            full_data = statement_data.copy()
            full_data['contract_id'] = contract_id
            full_data['article_num'] = art_num
            full_data['sentence_num'] = sentence_num
            full_data['statement_num'] = statement_num
            full_data['full_sentence'] = str(sent)
            # Note to self: statement_data contains a "full_statement"
            # key, so that gets "transferred" over to full_data.
            #print("Data to put in the db:")
            #print(data)
            statement_list.append(full_data)
    #print("Loop over sentences took " + str(total_pbs))
    return statement_list


def parse_by_subject(spacy_sent, resolve_corefs=True):
    """
    TODO: SMART THINGS like splitting the tree into *segments* such that each
    segment is the "phrase" for its subject. e.g. if sent has more than one subject:
    (1) find the HEAD subject
    (2) "cut off" all the other subtrees of *non-HEAD* subjects. In other words
        everywhere you see a subject, besides the HEAD, "clip" the tree at that
        node. Pull that node out of the tree and keep it separate
    (3) Iterating over all the subtrees of nodes in (2) gives you the non-HEAD
        phrases. Now to get the HEAD phrase you take all the tokens that haven't
        been "covered" yet, e.g., any token whose ONLY subject ancestor is HEAD.

    :param sent:
    :param resolve_corefs:
    :return:
    """
    #all_tokens = [t for t in sent]
    subjects = [t for t in spacy_sent if t.dep_ in subject_deps]
    all_statement_data = []
    # Each subject corresponds to a statement that it is the subject of.
    # Hence this is a loop over *statements*
    for subject_num, cur_subject in enumerate(subjects):
        cur_subdep = cur_subject.dep_
        # Again, debugging
        #if str(subject) == "Board" or str(subject) == "claim":
        #    print(subject)
        #    print(subject.head)
        #    print(list(subject.head.subtree))
        modal_lemma = None
        verb = cur_subject.head
        if not verb.tag_.startswith('V'):
            continue
        verb_lemma = verb.lemma_
        token_lists = defaultdict(list)
        neg = ''
        for vc_token in verb.children:
            if vc_token.tag_ == 'MD':
                modal_lemma = vc_token.orth_.lower()
                continue
            cur_tdep = vc_token.dep_
            if cur_tdep in ['punct', 'cc', 'det', 'meta', 'intj', 'dep']:
                continue
            if cur_tdep == 'neg':
                neg = 'not'
                #elif t.dep_ == 'auxpass':
            #    vlem = t.orth_.lower() + '_' + vlem
            elif cur_tdep == 'prt':
                verb_lemma = verb_lemma + '_' + vc_token.orth_.lower()
                #elif dep in maindeps:
            #    tokenlists[dep].append(t)
            else:
                #pass
                #print([modal,vlem,t,t.dep_,sent])
                #dcount[t.dep_] += 1
                token_lists[cur_tdep].append(vc_token)
        subject_lemma = cur_subject.lemma_
        #print("subject lemma: " + str(slem))
        in_coref = False
        cr_subject = cur_subject
        cr_slem = subject_lemma
        num_clusters = 0
        coref_replaced = False
        if resolve_corefs:
            in_coref = cur_subject._.in_coref
            # Now check if it's *different* from the coref cluster's main coref
            # TODO: Right now we take the first cluster. Instead, take the cluster
            # with the *closest* main to the subject
            if in_coref:
                coref_clusters = cur_subject._.coref_clusters
                num_clusters = len(coref_clusters)
                first_cluster = coref_clusters[0]
                # Get the main of this first cluster
                cluster_main_lem = first_cluster.main.lemma_
                if subject_lemma != cluster_main_lem:
                    # Replace it with main!
                    cr_slem = cluster_main_lem
                    coref_replaced = True

        statement_data = {
            'orig_subject': cur_subject.text,
            'orig_slem': subject_lemma,
            'in_coref': in_coref,
            'subject': cr_subject.text,
            'slem': cr_slem,
            'coref_replaced': coref_replaced,
            'modal': modal_lemma,
            'neg': neg,
            'verb': verb_lemma,
            #'full_sentence': str(sent),
            #'subfilter': 0,
            'passive': 0,
            'md': 0
        }
        if cur_subdep == 'nsubjpass':
            statement_data['passive'] = 1
        if modal_lemma is not None:
            statement_data['md'] = 1

        subject_phrase, subject_tags = get_branch(cur_subject, spacy_sent)

        statement_data['subject_branch'] = subject_phrase
        statement_data['subject_tags'] = subject_tags
        object_branches = []
        object_tags = []
        for tl_dep, tl_tokens in token_lists.items():
            if tl_dep in subject_deps:
                continue
            for tl_token in tl_tokens:
                tl_token_branch, tl_token_tags = get_branch(tl_token, spacy_sent)
                object_branches.append(tl_token_branch)
                object_tags.append(tl_token_tags)
        statement_data['object_branches'] = object_branches
        statement_data['object_tags'] = object_tags

        # Last but not least, the full text of the statement
        # (if possible?) TODO. It's NOT trivial. So for now it's
        # just always the empty string
        statement_data['full_statement'] = ""

        # So upon being added to datalist, the "data" dictionary has the following
        # keys: 'orig_subject','orig_slem','in_coref','subject', 'slem',"modal",
        # "neg","verb","passive","md","subject_branch","subject_tags",
        # "object_branches", "object_tags", "full_statement" (empty string for now)
        all_statement_data.append(statement_data)
    return all_statement_data


def recurse_tokens(*tokens):
    children = []
    def add(tok):
        sub = tok.children
        for item in sub:
            children.append(item)
            add(item)
    for token in tokens:
        add(token)
    return children
