import numpy as np
import pandas as pd
from scipy.sparse import find
import math
from numpy.ctypeslib import ndpointer
import ctypes
import utils
from config import *
from topicmodel import TopicModel
import topicmerge
import textvectorizer
from time import time

sellib = ctypes.CDLL('d:/projects/cpp/indeclib/x64/Release/indeclib.dll')
sellib.get_log_probs.argtypes = [
    ndpointer(ctypes.c_float), ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32), ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32),
    ctypes.c_int32, ctypes.c_float, ctypes.c_float
]


class TOPDMM:
    def __init__(self, n_topics, n_iter, alpha=0.1, beta=0.1, random_state=None, n_top=-1):
        self.n_topics = n_topics
        self.n_iter = n_iter
        if random_state is not None:
            np.random.seed(random_state)
        self.alpha = alpha
        self.beta = beta
        self.topic_word_ = None
        self.n_words = 0
        self.n_top = n_top

    def fit(self, X):
        D, self.n_words = X.shape
        K = self.n_topics

        N_d = np.asarray(X.sum(axis=1), np.int32).flatten()
        print(len(N_d), 'docs')

        # initialization
        N_k = np.zeros(K, np.int32)
        M_k = np.zeros(K, np.int32)
        # N_k_w = lil_matrix((K, self.n_words), dtype=np.int32)
        N_k_w = np.zeros((K, self.n_words), np.int32)

        K_d = np.zeros(D, np.int32)

        for d in range(D):
            k = np.random.choice(K, 1, p=[1.0 / K] * K)[0]
            K_d[d] = k
            M_k[k] = M_k[k] + 1
            N_k[k] = N_k[k] + N_d[d]
            x_tmp = X[d]
            for w, cnt in zip(x_tmp.indices, x_tmp.data):
                N_k_w[k, w] = N_k_w[k, w] + cnt

        for it in range(self.n_iter):
            # st = 0
            # t = time()
            for d in range(D):
                x = X[d]

                k_old = K_d[d]
                M_k[k_old] -= 1
                N_k[k_old] -= N_d[d]
                for w, cnt in zip(x.indices, x.data):
                    N_k_w[k_old, w] -= cnt
                # sample k_new
                probs = self.__calc_probs(N_k_w, x, N_d[d], M_k, N_k)
                k_new = np.random.choice(K, 1, p=probs)[0]
                K_d[d] = k_new
                M_k[k_new] += 1
                N_k[k_new] += N_d[d]
                for w, cnt in zip(x.indices, x.data):
                    N_k_w[k_new, w] += cnt
            # print(time() - t, st)
            if it % 5 == 0:
                print('iter {}, perplexity={}'.format(it, self.__perplexity(X, N_k_w, N_d, M_k, N_k)))
        self.topic_word_ = N_k_w + self.beta

    def __calc_probs(self, N_k_w, x, N_dd, M_k, N_k):
        log_probs = np.zeros(self.n_topics, np.float32)
        if self.n_top > -1:
            N_k_w_top = self.__get_N_k_w_top(N_k_w)
            sellib.get_log_probs(log_probs, x.indices, x.data, len(x.indices), N_dd, self.n_words,
                                 M_k, N_k, N_k_w_top, self.n_topics, self.alpha, self.beta)
        else:
            sellib.get_log_probs(log_probs, x.indices, x.data, len(x.indices), N_dd, self.n_words,
                                 M_k, N_k, N_k_w, self.n_topics, self.alpha, self.beta)
        # log_probs = self.__update_log_probs(log_probs, X[d].indices, X[d].data, N_d[d], V, M_k, N_k, N_k_w)
        # print(log_probs)
        probs = np.exp(log_probs)
        return probs / np.sum(probs)

    def __get_N_k_w_top(self, N_k_w):
        N_k_w_top = np.zeros((self.n_topics, self.n_words), np.int32)
        for k in range(self.n_topics):
            top_words = np.argpartition(-N_k_w[k], range(self.n_top))[:self.n_top]
            for w in top_words:
                N_k_w_top[k][w] = N_k_w[k][w]
        return N_k_w_top

    def __update_log_probs(self, log_probs, words, tfs, n_doc_words, n_vocab_words, M_k, N_k, N_k_w):
        for k in range(self.n_topics):
            log_probs[k] += math.log(self.alpha + M_k[k])
            for i, w in enumerate(words):
                for j in range(tfs[1]):
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

    def __perplexity(self, X, N_k_w, N_d, M_k, N_k):
        ll = 0.0
        for d, x in enumerate(X):
            probs = self.__calc_probs(N_k_w, x, N_d[d], M_k, N_k)
            p_max = np.max(probs)
            ll += np.log(p_max)
        return ll
        # return np.exp(ll / (-n))


def __show_coherences(k, topics, D_codoc):
    M_coh = 10
    cohs = list()
    coh_arr = np.zeros((k, k), np.float32)
    for i, t1 in enumerate(topics):
        c = TopicModel.coherence(t1, D_codoc, M_coh)
        cohs.append(c)
        widxs1 = np.argpartition(-t1, range(M_coh))[:M_coh]
        for j in range(i + 1, k):
            t2 = topics[j]
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
    # D_codoc = utils.get_codoc_matrix(cv.vocab, contents)

    # k = 3
    n_topic_words_disp = 10
    for k in range(10, 11):
        dmm = TOPDMM(k, 100, alpha=0.01, beta=0.01)
        dmm.fit(X)
        for t in dmm.topic_word_:
            widxs = np.argpartition(-t, range(n_topic_words_disp))[:n_topic_words_disp]
            topic_words = [cv.vocab[i] for i in widxs]
            print(' '.join(topic_words))

        # __show_coherences(k, dmm.topic_word_, D_codoc)

        test_vocab_file = os.path.join(QUORA_DATA_DIR, '{}_vocab.txt'.format(name))
        test_topic_file = os.path.join(QUORA_DATA_DIR, '{}_topics.txt'.format(name))
        dmm.save(cv.vocab, test_vocab_file, test_topic_file)


def __topdmm_wc(name, dst_vocab_file, dst_topics_file):
    all_doc_contents = utils.read_lines_to_list(WC_SEG_DOC_CONTENT_NODUP_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(WC_NAME_DOC_ND_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    print(len(contents), 'docs')

    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    extra_exclude_words = {name}
    if name == '姜子牙':
        extra_exclude_words = {'姜', '子牙'}
    cv = textvectorizer.CountVectorizer((WC_DF_ND_FILE, 20, 700), remove_stopwords=True, words_exist=words_exist,
                                        extra_exclude_words=extra_exclude_words)
    print(len(cv.vocab), 'words in vocab')
    X = cv.get_vecs(contents, normalize=False)
    # D_codoc = utils.get_codoc_matrix(cv.vocab, contents)

    n_topic_words_disp = 10
    print('starting training ...')
    # for k in range(10, 11):
    k = 10
    dmm = TOPDMM(k, 100, alpha=0.01, beta=0.01, n_top=-1)
    dmm.fit(X)
    for t in dmm.topic_word_:
        widxs = np.argpartition(-t, range(n_topic_words_disp))[:n_topic_words_disp]
        topic_words = [cv.vocab[i] for i in widxs]
        print(' '.join(topic_words))

    dmm.save(cv.vocab, dst_vocab_file, dst_topics_file)


def __run_with_wc():
    df = pd.read_csv(WC_ENTITY_NAMES_FILE, header=None)
    for ch_name, en_name in df.itertuples(False, None):
        # if en_name != 'cc':
        #     continue
        dst_vocab_file = os.path.join(WC_DATADIR, '{}_vocab.txt'.format(en_name))
        dst_topic_file = os.path.join(WC_DATADIR, '{}_topics.txt'.format(en_name))
        __topdmm_wc(ch_name, dst_vocab_file, dst_topic_file)
        # break

    # all_doc_contents = utils.read_lines_to_list(WC_SEG_DOC_CONTENT_NODUP_FILE)
    # name_doc_dict = utils.load_entity_name_to_doc_file(WC_NAME_DOC_ND_FILE)
    # doc_idxs = name_doc_dict[name]
    # contents = [all_doc_contents[idx] for idx in doc_idxs]
    # print(len(contents), 'docs')
    #
    # docs_words = [content.split(' ') for content in contents]
    # words_exist = utils.get_word_set(docs_words)
    # cv = textvectorizer.CountVectorizer((WC_DF_ND_FILE, 20, 700), remove_stopwords=True, words_exist=words_exist)
    # print(len(cv.vocab), 'words in vocab')
    # X = cv.get_vecs(contents, normalize=False)
    # # D_codoc = utils.get_codoc_matrix(cv.vocab, contents)
    #
    # n_topic_words_disp = 10
    # print('starting training ...')
    # # for k in range(10, 11):
    # k = 10
    # dmm = TOPDMM(k, 100, alpha=0.01, beta=0.01, n_top=10)
    # dmm.fit(X)
    # for t in dmm.topic_word_:
    #     widxs = np.argpartition(-t, range(n_topic_words_disp))[:n_topic_words_disp]
    #     topic_words = [cv.vocab[i] for i in widxs]
    #     print(' '.join(topic_words))
    #
    # test_vocab_file = os.path.join(WC_DATADIR, '{}_vocab.txt'.format(doc_name_dict[name]))
    # test_topic_file = os.path.join(WC_DATADIR, '{}_topics.txt'.format(doc_name_dict[name]))
    # dmm.save(cv.vocab, test_vocab_file, test_topic_file)


if __name__ == '__main__':
    # __run_with_quora()
    __run_with_wc()
