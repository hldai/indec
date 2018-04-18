import numpy as np
import time
import textvectorizer
from config import *
import utils


class UrnLDA:
    def __init__(self, alpha=5.0, beta=0.1, n_iter=50, k=10):
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.Z = list()
        self.k = k
        self.n_docs = 0
        self.n_words = 0
        self.ndz = None
        self.nzw = None
        self.nz = None
        self.A = None
        self.p_co = None
        self.vocab = None

    def fit(self, docs, vocab, word_idfs):
        self.n_docs = len(docs)
        self.n_words = len(vocab)
        self.vocab = vocab
        self.ndz = np.zeros([self.n_docs, self.k]) + self.alpha
        self.nzw = np.zeros([self.k, self.n_words]) + self.beta
        self.nz = np.zeros([self.k]) + self.n_words * self.beta

        self.__init_with_data(docs, word_idfs)

        for i in range(0, self.n_iter):
            self.__gibbs_sampling(docs)
            print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", self.__perplexity(docs))

        n_topic_words_disp = 10
        for z in range(self.k):
            top_widxs = np.argpartition(-self.nzw[z, :], range(n_topic_words_disp))[:n_topic_words_disp]
            pw = self.nzw[z, :] / np.sum(self.nzw[z, :])
            # print([vocab[w] for w in top_widxs])
            for w in top_widxs:
                print('{:.6f}*{}'.format(pw[w], self.vocab[w]), end='\t')
            print()

        print()
        print()
        self.__coh_assess()

    def save(self, dst_vocab_file, dst_topic_file):
        with open(dst_vocab_file, 'w', encoding='utf-8', newline='\n') as fout:
            for w in self.vocab:
                fout.write('{}\n'.format(w))
        with open(dst_topic_file, 'w', encoding='utf-8', newline='\n') as fout:
            for z in range(self.k):
                fout.write('{}\n'.format(' '.join([str(n) for n in self.nzw[z]])))

    def __coh_assess(self):
        n_top = 10
        coh_topics = np.zeros((self.k, self.k), np.float32)
        for z1 in range(self.k):
            pw1 = self.nzw[z1, :] / np.sum(self.nzw[z1, :])
            top_words1 = np.argpartition(-self.nzw[z1, :], range(n_top))[:n_top]
            for z2 in range(self.k):
                pw2 = self.nzw[z2, :] / np.sum(self.nzw[z2, :])
                top_words2 = np.argpartition(-self.nzw[z2, :], range(n_top))[:n_top]
                coh_topics[z1][z2] = self.__calc_topic_coh(top_words1, pw1, top_words2, pw2)
        for r in coh_topics:
            print(' '.join(['{:.8f}'.format(float(v)) for v in r]))

    def __calc_topic_coh(self, top_words1, pw1, top_words2, pw2):
        p = 0
        for w1 in top_words1:
            for w2 in top_words2:
                p += pw1[w1] * pw2[w2] * self.p_co[w1][w2]
        return p

    def __gibbs_sampling(self, docs):
        for d, doc_words in enumerate(docs):
            for pos, w in enumerate(doc_words):
                z = self.Z[d][pos]

                self.ndz[d, z] -= 1
                self.nzw[z, :] -= self.A[:, w]
                # self.nzw[z, w] -= 1
                self.nz[z] -= 1

                pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                # print(pz, pz.sum())
                z = np.random.multinomial(1, pz / pz.sum()).argmax()
                self.Z[d][pos] = z

                self.ndz[d, z] += 1
                self.nzw[z, :] += self.A[:, w]
                # self.nzw[z, w] += 1
                self.nz[z] += 1

    def __init_with_data(self, docs, word_idfs):
        self.__init_A(docs, word_idfs)
        for d, doc in enumerate(docs):
            z_curdoc = []
            for w in doc:
                pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                z_tmp = np.random.multinomial(1, pz / pz.sum()).argmax()
                z_curdoc.append(z_tmp)
                self.ndz[d, z_tmp] += 1
                for v in range(self.n_words):
                    self.nzw[z_tmp, v] += self.A[v, w]
                # self.nzw[z_tmp, w] += 1
                self.nz[z_tmp] += 1
            self.Z.append(z_curdoc)

    def __init_A(self, docs, word_idfs):
        n_docs = len(docs)
        self.p_co = np.zeros((self.n_words, self.n_words), np.float32)
        word_docs = [set() for _ in range(self.n_words)]
        for i, doc in enumerate(docs):
            words = set(doc)
            for w in words:
                word_docs[w].add(i)

        self.A = np.zeros((self.n_words, self.n_words), np.float32)
        # lambda_v = np.zeros(self.n_words, np.float32)
        for w1 in range(self.n_words):
            docs1 = word_docs[w1]
            # print(docs1)
            # lambda_v[w1] = np.log(n_docs / len(docs1))
            self.A[w1][w1] = len(docs1)
            for w2 in range(w1 + 1, self.n_words):
                if w1 == w2:
                    continue
                docs2 = word_docs[w2]
                si = docs1.intersection(docs2)
                su = docs1.union(docs2)
                self.A[w1][w2] = self.A[w2][w1] = len(si)
                self.p_co[w1][w2] = self.p_co[w2][w1] = len(si) / len(su)

        # word_idfs = [(self.vocab[w], lambda_v[w]) for w in range(self.n_words)]
        # word_idfs.sort(key=lambda x: x[1])
        # for w, idf in word_idfs:
        #     print(w, idf)
        # exit()

        for j in range(self.n_words):
            for i in range(self.n_words):
                # self.A[i][j] *= lambda_v[i]
                self.A[i][j] *= word_idfs[i]
            sum_col = np.sum(self.A[:, j])
            for i in range(self.n_words):
                self.A[i][j] /= sum_col

    def __perplexity(self, docs):
        nd = np.sum(self.ndz, 1)
        n = 0
        ll = 0.0
        for d, doc in enumerate(docs):
            for w in doc:
                ll = ll + np.log(((self.nzw[:, w] / self.nz) * (self.ndz[d, :] / nd[d])).sum())
                n = n + 1
        return np.exp(ll / (-n))


def process_quora():
    name = 'DC'
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer(QUORA_DF_FILE, 50, 6000, remove_stopwords=True, words_exist=words_exist)
    print(len(cv.vocab), 'words in vocab')
    word_idfs = [np.log(QUORA_NUM_TOTAL_DOCS / cv.word_cnts[w]) for w in cv.vocab]

    docs = list()
    for words in docs_words:
        doc = list()
        for w in words:
            widx = cv.word_dict.get(w, -1)
            if widx > -1:
                doc.append(widx)
        docs.append(doc)

    return docs, cv.vocab, word_idfs


def __check_topics():
    k = 10
    n_top = 10

    vocab = utils.read_lines_to_list(test_vocab_file)
    n_words = len(vocab)
    print(n_words, 'words')
    nzw = np.zeros((k, n_words), np.float32)

    # coh_topics = np.zeros((k, k), np.float32)
    # for z1 in range(k):
    #     pw1 = nzw[z1, :] / np.sum(nzw[z1, :])
    #     top_words1 = np.argpartition(-nzw[z1, :], range(n_top))[:n_top]
    #     for z2 in range(k):
    #         pw2 = nzw[z2, :] / np.sum(nzw[z2, :])
    #         top_words2 = np.argpartition(-nzw[z2, :], range(n_top))[:n_top]
    #         coh_topics[z1][z2] = __calc_topic_coh(top_words1, pw1, top_words2, pw2)
    # for r in coh_topics:
    #     print(' '.join(['{:.8f}'.format(float(v)) for v in r]))


urnlda = UrnLDA(alpha=0.1, n_iter=50, k=10)
docs, vocab, word_idfs = process_quora()
urnlda.fit(docs, vocab, word_idfs)

test_vocab_file = os.path.join(QUORA_DATA_DIR, 'dc_vocab.txt')
test_topic_file = os.path.join(QUORA_DATA_DIR, 'dc_topics.txt')
urnlda.save(test_vocab_file, test_topic_file)
