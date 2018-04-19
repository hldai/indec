import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import find
import math
import utils
from config import *
import textvectorizer


class GSDMM:
    def __init__(self, n_topics, n_iter, alpha=0.1, beta=0.1, random_state=None):
        self.n_topics = n_topics
        self.n_iter = n_iter
        if random_state is not None:
            np.random.seed(random_state)
        self.alpha = alpha
        self.beta = beta
        self.topic_word_ = None

    def fit(self, X):
        alpha = self.alpha
        beta = self.beta

        D, V = X.shape
        K = self.n_topics

        N_d = np.asarray(X.sum(axis=1), np.int32).flatten()
        print(len(N_d), 'docs')
        words_d = {}
        for d in range(D):
            words_d[d] = find(X[d, :])[1]

        # initialization
        N_k = np.zeros(K)
        M_k = np.zeros(K)
        # N_k_w = lil_matrix((K, V), dtype=np.int32)
        N_k_w = np.zeros((K, V), np.int32)

        K_d = np.zeros(D, np.int32)

        for d in range(D):
            k = np.random.choice(K, 1, p=[1.0 / K] * K)[0]
            K_d[d] = k
            M_k[k] = M_k[k] + 1
            N_k[k] = N_k[k] + N_d[d]
            for w in words_d[d]:
                N_k_w[k, w] = N_k_w[k, w] + X[d, w]

        for iter in range(self.n_iter):
            print('iter ', iter)
            for d in range(D):
                k_old = K_d[d]
                M_k[k_old] -= 1
                N_k[k_old] -= N_d[d]
                for w in words_d[d]:
                    N_k_w[k_old, w] -= X[d, w]
                # sample k_new
                log_probs = [0] * K
                for k in range(K):
                    log_probs[k] += math.log(alpha + M_k[k])
                    for w in words_d[d]:
                        N_d_w = X[d, w]
                        for j in range(N_d_w):
                            log_probs[k] += math.log(N_k_w[k, w] + beta + j)
                    for i in range(N_d[d]):
                        log_probs[k] -= math.log(N_k[k] + beta * V + i)
                log_probs = np.array(log_probs) - max(log_probs)
                probs = np.exp(log_probs)
                probs = probs / np.sum(probs)
                k_new = np.random.choice(K, 1, p=probs)[0]
                K_d[d] = k_new
                M_k[k_new] += 1
                N_k[k_new] += N_d[d]
                for w in words_d[d]:
                    N_k_w[k_new, w] += X[d, w]
        self.topic_word_ = N_k_w


if __name__ == '__main__':
    # name = 'DC'
    name = 'WP'
    # name = 'Austin'
    # name = 'Mark'
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer(QUORA_DF_FILE, 50, 6000, remove_stopwords=True, words_exist=words_exist)
    print(len(cv.vocab), 'words in vocab')
    X = cv.get_vecs(contents, normalize=False)

    k = 10
    dmm = GSDMM(k, 50)
    # print(X.astype())
    # exit()
    dmm.fit(X)
    for t in dmm.topic_word_:
        widxs = np.argpartition(-t, range(10))[:10]
        topic_words = [cv.vocab[i] for i in widxs]
        print(' '.join(topic_words))
