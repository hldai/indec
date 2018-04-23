import numpy as np
import pandas as pd
from scipy.sparse import find
import math
import utils
from config import *
from topicmodel import TopicModel
import topicmerge
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
        M_k = np.zeros(K, np.int32)
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
                log_probs = np.zeros(K, np.float32)
                log_probs = self.__update_log_probs(log_probs, words_d[d], M_k, X[d], N_k_w, N_d[d], N_k, V)
                probs = np.exp(log_probs)
                probs = probs / np.sum(probs)
                k_new = np.random.choice(K, 1, p=probs)[0]
                K_d[d] = k_new
                M_k[k_new] += 1
                N_k[k_new] += N_d[d]
                for w in words_d[d]:
                    N_k_w[k_new, w] += X[d, w]
        self.topic_word_ = N_k_w + self.beta

    def __update_log_probs(self, log_probs, words_cur_doc, M_k, count_vec, N_k_w, n_doc_words, N_k, n_vocab_words):
        for k in range(self.n_topics):
            log_probs[k] += math.log(self.alpha + M_k[k])
            for w in words_cur_doc:
                N_d_w = count_vec[w]
                for j in range(N_d_w):
                    log_probs[k] += math.log(N_k_w[k, w] + self.beta + j)
            for i in range(n_doc_words):
                log_probs[k] -= math.log(N_k[k] + self.beta * n_vocab_words + i)
        return np.array(log_probs) - max(log_probs)

    def save(self, vocab, vocab_file, topic_file):
        utils.write_list_to_lines(vocab, vocab_file)
        with open(topic_file, 'w', encoding='utf-8', newline='\n') as fout:
            pd.DataFrame(self.topic_word_).to_csv(fout, header=False, index=False)
            # for t in self.topic_word_:
            #     fout.write('{}\n'.format(' '.join([str(n) for n in t])))


def __run_with_quora():
    name = 'DC'
    # name = 'WP'
    # name = 'Austin'
    # name = 'Mark'
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer((QUORA_DF_FILE, 50, 6000), remove_stopwords=True, words_exist=words_exist)
    print(len(cv.vocab), 'words in vocab')
    X = cv.get_vecs(contents, normalize=False)
    D_codoc = utils.get_codoc_matrix(cv.vocab, contents)

    # k = 3
    for k in range(10, 11):
        dmm = GSDMM(k, 100, alpha=0.01, beta=0.01)
        dmm.fit(X)
        for t in dmm.topic_word_:
            widxs = np.argpartition(-t, range(10))[:10]
            topic_words = [cv.vocab[i] for i in widxs]
            print(' '.join(topic_words))

        M_coh = 10
        cohs = list()
        coh_arr = np.zeros((k, k), np.float32)
        for i, t1 in enumerate(dmm.topic_word_):
            c = TopicModel.coherence(t1, D_codoc, M_coh)
            cohs.append(c)
            widxs1 = np.argpartition(-t1, range(M_coh))[:M_coh]
            for j in range(i + 1, k):
                t2 = dmm.topic_word_[j]
                t_tmp = t1 + t2
                c_tmp = TopicModel.coherence(t_tmp, D_codoc, M_coh)
                widxs2 = np.argpartition(-t2, range(M_coh))[:M_coh]
                for w1 in widxs1:
                    for w2 in widxs2:
                        coh_arr[i][j] += D_codoc[w1][w2] / D_codoc[w1][w1] / D_codoc[w2][w2]
                coh_arr[j][i] = coh_arr[i][j]
        print(cohs)
        for cs in coh_arr:
            print(' '.join([str(v) for v in cs]))

        test_vocab_file = os.path.join(QUORA_DATA_DIR, '{}_vocab.txt'.format(name))
        test_topic_file = os.path.join(QUORA_DATA_DIR, '{}_topics.txt'.format(name))
        dmm.save(cv.vocab, test_vocab_file, test_topic_file)


def __run_with_wc():
    name = '曹操'

    all_doc_contents = utils.read_lines_to_list(WC_SEG_DOC_CONTENT_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(WC_NAME_DOC_FILE)
    doc_idxs = name_doc_dict['曹操']
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    print(len(contents), 'docs')

    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer((WC_DF_FILE, 50, 3000), remove_stopwords=True, words_exist=words_exist)
    print(len(cv.vocab), 'words in vocab')
    X = cv.get_vecs(contents, normalize=False)
    # D_codoc = utils.get_codoc_matrix(cv.vocab, contents)

    print('starting training ...')
    for k in range(10, 11):
        dmm = GSDMM(k, 100, alpha=0.01, beta=0.01)
        dmm.fit(X)
        for t in dmm.topic_word_:
            widxs = np.argpartition(-t, range(10))[:10]
            topic_words = [cv.vocab[i] for i in widxs]
            print(' '.join(topic_words))

        # M_coh = 10
        # cohs = list()
        # coh_arr = np.zeros((k, k), np.float32)
        # for i, t1 in enumerate(dmm.topic_word_):
        #     c = TopicModel.coherence(t1, D_codoc, M_coh)
        #     cohs.append(c)
        #     widxs1 = np.argpartition(-t1, range(M_coh))[:M_coh]
        #     for j in range(i + 1, k):
        #         t2 = dmm.topic_word_[j]
        #         t_tmp = t1 + t2
        #         c_tmp = TopicModel.coherence(t_tmp, D_codoc, M_coh)
        #         widxs2 = np.argpartition(-t2, range(M_coh))[:M_coh]
        #         for w1 in widxs1:
        #             for w2 in widxs2:
        #                 coh_arr[i][j] += D_codoc[w1][w2] / D_codoc[w1][w1] / D_codoc[w2][w2]
        #         coh_arr[j][i] = coh_arr[i][j]
        # print(cohs)
        # for cs in coh_arr:
        #     print(' '.join([str(v) for v in cs]))

        test_vocab_file = os.path.join(QUORA_DATA_DIR, '{}_vocab.txt'.format(name))
        test_topic_file = os.path.join(QUORA_DATA_DIR, '{}_topics.txt'.format(name))
        dmm.save(cv.vocab, test_vocab_file, test_topic_file)


if __name__ == '__main__':
    # __run_with_quora()
    __run_with_wc()
