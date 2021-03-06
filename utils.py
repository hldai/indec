import json


def load_entity_name_to_doc_file(filename):
    name_doc_dict = dict()
    f = open(filename, encoding='utf-8')
    for line in f:
        obj = json.loads(line)
        name_doc_dict[obj['entity_name']] = obj['docs']
    f.close()
    return name_doc_dict


def load_entity_name_to_minidoc_file(minidoc_info_file):
    import pandas as pd

    name_doc_dict = dict()
    df = pd.read_csv(minidoc_info_file)
    for mdid, doc_id, entity_name in df.itertuples(False, None):
        mdids = name_doc_dict.get(entity_name, list())
        if not mdids:
            name_doc_dict[entity_name] = mdids
        mdids.append(mdid)
    return name_doc_dict


def get_word_set(docs):
    words = set()
    for doc in docs:
        for w in doc:
            words.add(w)
    return words


def get_codoc_matrix(vocab, text_list):
    import numpy as np

    n_words = len(vocab)
    word_dict = {w: i for i, w in enumerate(vocab)}
    D = np.zeros((n_words, n_words), np.int32)
    for text in text_list:
        words = set(text.strip().split(' '))
        words = [word_dict[w] for w in words if w in word_dict]
        l = len(words)
        for i in range(l):
            w1 = words[i]
            for j in range(i, l):
                w2 = words[j]
                D[w1][w2] += 1
    return D


def load_docs_with_name(name, docs_file, name_doc_file):
    all_doc_contents = read_lines_to_list(docs_file)
    name_doc_dict = load_entity_name_to_doc_file(name_doc_file)
    doc_idxs = name_doc_dict[name]
    return [all_doc_contents[idx] for idx in doc_idxs]


def disp_topics(vocab, topics, n_words=10):
    import numpy as np
    for t in topics:
        idxs = np.argpartition(-t, range(n_words))[:n_words]
        print(' '.join([vocab[i] for i in idxs]))


def remove_lines(src_file, lines, dst_file):
    f = open(src_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        if i in lines:
            continue
        fout.write(line)
    f.close()
    fout.close()


def load_json_objs(filename):
    with open(filename, encoding='utf-8') as f:
        objs = [json.loads(line) for line in f]
    return objs
