import numpy as np
import scipy
import math
import utils
import textvectorizer
from config import *


class DPMFS:
    def __init__(self, n_words, N, n_docs):
        self.n_words = n_words
        self.n_docs = n_docs
        self.gamma = np.random.randint(0, 2, n_words, np.int32)
        self.z = np.random.randint(1, N + 1, n_docs, np.int32)
        self.omega = 0.1
        self.lamb = np.ones(n_words, np.float32) * 0.1
        self.alpha = 1.0
        self.r1 = 5
        self.r2 = 5

    def fit(self, X):
        Ti_log_sum = 0
        for i, x in enumerate(X):
            s_x, s_xgamma1, s_xgamma2 = 0, 0, 0
            # print(' '.join([str(j) for j in x.indices]))
            for j, v in zip(x.indices, x.data):
                s_xgamma1 += v * self.gamma[j]
                s_xgamma2 += v * (1 - self.gamma[j])
                s_x += v
            Ti_log = utils.log_factorial(s_xgamma1) + utils.log_factorial(s_xgamma2) - utils.log_factorial(s_x)
            Ti_log_sum += Ti_log

        tmp1, tmp2 = 0, 0
        for j in range(self.n_words):
            tmp1 += self.lamb[j] * (1 - self.gamma[j])
        for i, x in enumerate(X):
            for j, v in zip(x.indices, x.data):
                tmp2 += v * (1 - self.gamma[j])
        tmp2 += tmp1
        S1_log = scipy.special.loggamma(tmp1) - scipy.special.loggamma(tmp2)
        print(Ti_log_sum, S1_log)

        S2_log = 0
        for j in range(self.n_words):
            if self.gamma[j] == 1:
                continue
            tmp1 = 0
            # TODO


def __run_quora():
    name = 'DC'
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer((QUORA_DF_FILE, 100, 5000), remove_stopwords=True, words_exist=words_exist)
    print(len(cv.vocab), 'words in vocab')
    X = cv.get_vecs(contents)

    n_docs = len(doc_idxs)
    dpmfs = DPMFS(cv.n_words, 10, n_docs)
    dpmfs.fit(X)


if __name__ == '__main__':
    __run_quora()
