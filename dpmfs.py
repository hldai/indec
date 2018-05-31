import numpy as np
import scipy
import math
import utils
import textvectorizer
from config import *


class DPMFS:
    def __init__(self, n_words, N, n_docs):
        self.N = N
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
        Xt = X.transpose().tocsr()
        f_x = self.__log_likelihood_docs(X, Xt)
        p_gamma = self.__log_prior_gamma()
        print(f_x, p_gamma)

    def __log_prior_gamma(self):
        r = 0
        for j in range(self.n_words):
            r += self.gamma[j] * np.log(self.omega) + (1 - self.gamma[j]) * np.log(1 - self.omega)
        return r

    def __log_likelihood_docs(self, X, Xt):
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

        S2_log = 0
        for j, xj in enumerate(Xt):
            if self.gamma[j] == 1:
                continue
            S2_log += scipy.special.loggamma(np.sum(xj.data) + self.lamb[j])
            S2_log -= scipy.special.loggamma(self.lamb[j])

        Q_log = 0
        for j in range(self.n_words):
            Q_log += self.lamb[j] * self.gamma[j]
        Q_log = scipy.special.loggamma(Q_log)
        for j in range(self.n_words):
            if self.gamma[j] == 0:
                continue
            Q_log -= scipy.special.loggamma(self.gamma[j])

        Rk_sum_log = 0
        for k in range(self.N):
            Rk_log = 0
            for j, xj in enumerate(Xt):
                if self.gamma[j] == 1:
                    continue
                tmp1 = 0
                for i, v in zip(xj.indices, xj.data):
                    if self.z[i] == k:
                        tmp1 += v
                tmp1 += self.lamb[j]
                Rk_log += scipy.special.loggamma(tmp1)
            tmp2 = 0
            for i, xi in enumerate(X):
                if self.z[i] != k:
                    continue
                for j, xij in zip(xi.indices, xi.data):
                    tmp2 += xij * self.gamma[j]
            for j in range(self.n_words):
                tmp2 += self.lamb[j] * self.gamma[j]
            Rk_log -= scipy.special.loggamma(tmp2)
            Rk_sum_log += Rk_log

        r = Ti_log_sum + S1_log + S2_log + Q_log + Rk_sum_log
        return r.real


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
