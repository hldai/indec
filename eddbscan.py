import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import textvectorizer
from config import *
import utils


def __load_entity_name_to_doc():
    name_doc_dict = dict()
    f = open(NAME_DOC_FILE, encoding='utf-8')
    for line in f:
        obj = json.loads(line)
        name_doc_dict[obj['entity_name']] = obj['docs']
    f.close()
    return name_doc_dict


def __check_dists(x, contents):
    k = 10
    # idxs = np.argpartition(x[0], np.arange(k))
    # for idx in idxs[:k]:
    #     print(x[0][idx])
    #     print(contents[idx])
    #     print()

    idxs = np.argpartition(-x[0], np.arange(k))
    for idx in idxs[:k]:
        print(x[0][idx])
        print(contents[idx])
        print()
    # print(x[0][1])
    # print(x[0][7])
    # print(x[0][186])
    # print(x[0][1450])
    # print(x[0][1708])


def __entity_disamb():
    tv = textvectorizer.TfIdf(DF_FILE, 5, 62000, 14357)
    all_doc_contents = utils.read_lines_to_list(SEG_DOC_CONTENT_FILE)

    eps = 0.55
    min_samples = 20

    name_doc_dict = __load_entity_name_to_doc()
    for name, doc_idxs in name_doc_dict.items():
        print(name)
        contents = [all_doc_contents[idx] for idx in doc_idxs]
        vecs = tv.get_vecs(contents)
        X = cosine_distances(vecs)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        dbscan.fit(X)
        print(X[0][1708])
        print(dbscan.labels_[1708])
        l_min, l_max = min(dbscan.labels_), max(dbscan.labels_)
        n_clusters = l_max - l_min
        print(n_clusters, 'clusters')
        # if n_clusters > 15:
        #     break
        # __check_dists(x, contents)
        labels = set(dbscan.labels_)
        fouts_dict = {l: open(os.path.join(
            dbscan_result_dir, 'l{}.txt'.format(l)),
            'w', encoding='utf-8', newline='\n') for l in labels}
        for idx, l in enumerate(dbscan.labels_):
            fout = fouts_dict[l]
            fout.write('{}\n'.format(contents[idx].strip()))

        for fout in fouts_dict.values():
            fout.close()
        break


def __gen_docs_with_specific_name():
    all_doc_contents = utils.read_lines_to_list(DOC_CONTENT_FILE)
    name_doc_dict = __load_entity_name_to_doc()
    doc_idxs = name_doc_dict['曹操']
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    print(len(contents), 'docs')
    fout = open('d:/data/indec/cc.txt', 'w', encoding='utf-8', newline='\n')
    for text in contents:
        fout.write('{}\n'.format(text.strip()))
    fout.close()


dbscan_result_dir = os.path.join(DATADIR, 'dbscan')
# __gen_docs_with_specific_name()
__entity_disamb()
