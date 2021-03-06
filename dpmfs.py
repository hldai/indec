import numpy as np
import scipy
import argparse
from collections import Counter
import utils
import textvectorizer
from config import *


class DPMFS:
    def __init__(self, n_words, N, n_docs, n_iter, lamb=0.5):
        self.N = N
        self.n_words = n_words
        self.n_docs = n_docs
        self.z = np.random.randint(1, N + 1, n_docs, np.int32)
        self.eta = np.zeros((self.N + 1, self.n_words), np.float32)
        self.omega = 0.05

        # self.gamma = np.random.randint(0, 2, n_words, np.int32)
        self.gamma = np.zeros(n_words, np.int32)
        rand_v = np.random.uniform(0, 1, n_words)
        for i, v in enumerate(rand_v):
            if v < self.omega:
                self.gamma[i] = 1

        self.lamb = np.ones(n_words, np.float32) * lamb
        self.alpha = 0.1
        self.r1 = 5
        self.r2 = 5
        self.n_iter = n_iter

    def fit(self, X, log_file=None, vocab=None):
        print('lamb={}, alpha={}, omega={}'.format(self.lamb[0], self.alpha, self.omega))
        Xt = X.transpose().tocsr()
        for it in range(self.n_iter):
            self.__update_gamma(X, Xt)
            self.__update_eta(X)
            self.__update_z(X)
            self.__update_lamb(Xt)
            # print(self.z)
            print(it, np.sum(self.gamma), Counter(self.z))
            if it % 10 == 0 and log_file is not None:
                print('save z to {}'.format(log_file))
                np.savetxt(log_file, self.z, fmt='%d')
                self.__print_topics(X, vocab)

    def __print_topics(self, X, vocab):
        topic_word_cnts = np.zeros((self.N + 1, self.n_words), np.int32)
        for i, (x, k) in enumerate(zip(X, self.z)):
            for w, v in zip(x.indices, x.data):
                if self.gamma[w] == 0:
                    continue
                # print(k, w)
                topic_word_cnts[k][w] += v

        for k in range(self.N):
            n_sum = np.sum(topic_word_cnts[k])
            if n_sum == 0:
                continue
            idxs = np.argpartition(-topic_word_cnts[k], range(10))[:10]
            words = [vocab[idx] for idx in idxs]
            print(k, ' '.join(words))

    def __update_lamb(self, Xt):
        gamma_indices_neg = [i for i, v in enumerate(self.gamma) if v == 0]
        w_idx_dict = {v: i for i, v in enumerate(gamma_indices_neg)}
        n_neg = len(gamma_indices_neg)

        dir_params = np.zeros(n_neg, np.float32)
        for i, l in enumerate(gamma_indices_neg):
            dir_params[i] = (1 - self.gamma[l]) * self.lamb[l]

        for l, xt in enumerate(Xt):
            if l not in w_idx_dict:
                continue
            sum_tmp = 0
            for i, xil in zip(xt.indices, xt.data):
                sum_tmp += xil
            dir_params[w_idx_dict[l]] += (1 - self.gamma[l]) * sum_tmp
        eta0_new = np.random.dirichlet(dir_params)
        for idx, v in zip(gamma_indices_neg, eta0_new):
            self.eta[0][idx] = v
        # self.eta[0] = eta0_new

    def __update_z(self, X):
        for i in range(self.n_docs):
            for _ in range(self.r2):
                params = np.zeros(self.N, np.float32)
                for k in range(1, self.N + 1):
                    niz = 0
                    for j in range(self.n_docs):
                        if j != i and self.z[j] == k:
                            niz += 1
                    params[k - 1] = (niz + self.alpha / self.N) / (self.n_docs - 1 + self.alpha)
                zi_new = np.random.multinomial(1, params)
                zi_new = zi_new.nonzero()[0][0] + 1
                if zi_new == self.z[i]:
                    continue

                x = X[i]
                f_xgamma_new = self.__f_xgamma(x, self.eta[zi_new])
                f_xgamma_cur = self.__f_xgamma(x, self.eta[self.z[i]])

                d = f_xgamma_new - f_xgamma_cur
                # print(zi_new, self.z[i], f_xgamma_new, f_xgamma_cur, np.exp(d))
                # for k in range(1, self.N + 1):
                #     tmp = self.__f_xgamma(x, self.eta[k])
                #     print(self.z[i], k, f_xgamma_cur, tmp, np.exp(tmp - f_xgamma_cur))

                if d > 0:
                    self.z[i] = zi_new
                else:
                    tmp_rand = np.random.uniform(0, 1)
                    if np.log(tmp_rand) < d:
                        self.z[i] = zi_new

    def __f_xgamma(self, x, eta):
        result = 0
        for j, xij in zip(x.indices, x.data):
            if self.gamma[j] == 0:
                continue
            if eta[j] == 0:
                result += -1e8
            else:
                result += xij * np.log(eta[j])
        return result

    def __update_eta(self, X):
        gamma_indices_pos = [i for i, v in enumerate(self.gamma) if v == 1]
        w_idx_dict = {idx: i for i, idx in enumerate(gamma_indices_pos)}
        n_pos = len(gamma_indices_pos)

        z_set = set(self.z)
        k_neg = [k for k in range(1, self.N + 1) if k not in z_set]
        for k in k_neg:
            eta_new = np.random.dirichlet([self.lamb[i] * self.gamma[i] for i in gamma_indices_pos])
            self.eta[k] = np.zeros(self.n_words)
            for idx, v in zip(gamma_indices_pos, eta_new):
                self.eta[k][idx] = v

        z_doc_dict = {k: list() for k in range(1, self.N + 1)}
        for i, zk in enumerate(self.z):
            z_doc_dict[zk].append(i)
        for k, docs in z_doc_dict.items():
            if not docs:
                continue

            params = np.zeros(n_pos, np.float32)
            for idx, l in enumerate(gamma_indices_pos):
                params[idx] = self.lamb[l] * self.gamma[l]
                # if params[idx] <= 0:
                #     print('ddd', params[idx])
            for j in docs:
                xj = X[j]
                for l, xjl in zip(xj.indices, xj.data):
                    if l not in w_idx_dict:
                        continue
                    params[w_idx_dict[l]] += xjl * self.gamma[l]
            # print(params)
            eta_new = np.random.dirichlet(params)
            self.eta[k] = np.zeros(self.n_words)
            for idx, v in zip(gamma_indices_pos, eta_new):
                self.eta[k][idx] = v
            # self.eta[k] = eta_new

    def __update_gamma(self, X, Xt):
        for i in range(self.r1):
            f_x = self.__log_likelihood_docs(X, Xt)
            p_gamma = self.__log_prior_gamma()
            f_gamma_old = f_x + p_gamma

            w = np.random.randint(0, self.n_words)
            self.gamma[w] = 0 if self.gamma[w] == 1 else 1
            f_x = self.__log_likelihood_docs(X, Xt)
            p_gamma = self.__log_prior_gamma()
            f_gamma_new = f_x + p_gamma

            p_accept = f_gamma_new - f_gamma_old
            if p_accept < 0:
                v = np.random.uniform(0, 1)
                if np.log(v) > p_accept:
                    self.gamma[w] = 0 if self.gamma[w] == 1 else 1

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
            for j, xij in zip(x.indices, x.data):
                s_xgamma1 += xij * self.gamma[j]
                s_xgamma2 += xij * (1 - self.gamma[j])
                s_x += utils.log_factorial(xij)
            # Ti_log = utils.log_factorial(s_xgamma1) + utils.log_factorial(s_xgamma2) - utils.log_factorial(s_x)
            Ti_log = utils.log_factorial(s_xgamma1) + utils.log_factorial(s_xgamma2) - s_x
            Ti_log_sum += Ti_log

        tmp1, tmp2 = 0, 0
        for j in range(self.n_words):
            tmp1 += self.lamb[j] * (1 - self.gamma[j])
        for i, x in enumerate(X):
            for j, xij in zip(x.indices, x.data):
                tmp2 += xij * (1 - self.gamma[j])
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
            Q_log -= scipy.special.loggamma(self.lamb[j])
        z_vals = set(self.z)
        Q_log = len(z_vals) * Q_log

        Rk_sum_log = 0
        for k in range(self.N):
            Rk_log = 0
            for j, xj in enumerate(Xt):
                if self.gamma[j] == 0:
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-lamb", "--lamb", type=float)
    parser.add_argument("-N", "--N", type=int)
    args = parser.parse_args()
    lamb = 1 if args.lamb is None else args.lamb
    N = 10 if args.N is None else args.N

    dst_file = os.path.join(QUORA_DATA_DIR, 'dpmfs_z_{}.txt'.format(lamb))
    print(dst_file)

    n_docs = len(doc_idxs)
    dpmfs = DPMFS(cv.n_words, N=N, n_docs=n_docs, n_iter=1000, lamb=lamb)
    dpmfs.fit(X, dst_file, cv.vocab)
    np.savetxt(dst_file, dpmfs.z, fmt='%d')


if __name__ == '__main__':
    __run_quora()
