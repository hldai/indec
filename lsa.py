from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import numpy as np
import utils
import textvectorizer
from config import *


def __process_quora():
    name = 'DC'
    # name = 'Mark'
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer(QUORA_DF_FILE, 50, 6000, remove_stopwords=True, words_exist=words_exist)
    print(len(cv.vocab), 'words in vocab')
    X = cv.get_vecs(contents, normalize=True)
    print(X.shape)

    k = 10
    tsvd = TruncatedSVD(n_components=k)
    X_new = tsvd.fit_transform(X)
    for i in range(k):
        max_idxs = np.argpartition(-tsvd.components_[i], range(20))[:20]
        words = [cv.vocab[idx] for idx in max_idxs]
        print(tsvd.explained_variance_[i], tsvd.singular_values_[i])
        print(words)


__process_quora()
