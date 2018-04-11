import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import textvectorizer
from config import *
import utils


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


def __dbscan_docs(contents, tfidf, eps, min_samples, result_dir):
    vecs = tfidf.get_vecs(contents)
    X = cosine_distances(vecs)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    dbscan.fit(X)
    # print(X)
    # print(X[0][1708])
    # print(dbscan.labels_[1708])
    l_min, l_max = min(dbscan.labels_), max(dbscan.labels_)
    n_clusters = l_max - l_min
    print(n_clusters, 'clusters')
    if n_clusters > 10:
        return
    # __check_dists(x, contents)
    labels = set(dbscan.labels_)
    fouts_dict = {l: open(os.path.join(
        result_dir, 'l{}.txt'.format(l)),
        'w', encoding='utf-8', newline='\n') for l in labels}
    for idx, l in enumerate(dbscan.labels_):
        fout = fouts_dict[l]
        fout.write('{}\n'.format(contents[idx].strip()))

    for fout in fouts_dict.values():
        fout.close()


def __process_wechat():
    tfidf = textvectorizer.TfIdf(WC_DF_FILE, 5, 62000, 14357)
    all_doc_contents = utils.read_lines_to_list(WC_SEG_DOC_CONTENT_FILE)

    eps = 0.55
    min_samples = 20

    name_doc_dict = utils.load_entity_name_to_doc_file(WC_NAME_DOC_FILE)
    for name, doc_idxs in name_doc_dict.items():
        print(name)
        contents = [all_doc_contents[idx] for idx in doc_idxs]
        __dbscan_docs(contents, tfidf, eps, min_samples, result_dir_wc)
        break


def __process_quora():
    tfidf = textvectorizer.TfIdf(QUORA_DF_FILE, 10, 24600, 143479, True)
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)

    eps = 0.7
    min_samples = 3

    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict['DC']
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    __dbscan_docs(contents, tfidf, eps, min_samples, result_dir_quora)


result_dir_wc = os.path.join(WC_DATADIR, 'dbscan')
result_dir_quora = os.path.join(QUORA_DATA_DIR, 'dbscan')
# __process_wechat()
__process_quora()
