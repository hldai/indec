import numpy as np
import time
import textvectorizer
from config import *
import utils


class UrnLDA:
    def __init__(self, alpha=5, beta=0.1, n_iter=50, k=10):
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

    def fit(self, docs, word_idx_dict, vocab):
        self.n_docs = len(docs)
        self.n_words = len(vocab)
        self.vocab = vocab
        self.ndz = np.zeros([self.n_docs, self.k]) + self.alpha
        self.nzw = np.zeros([self.k, self.n_words]) + self.beta
        self.nz = np.zeros([self.k]) + self.n_words * self.beta

        self.__init_with_data(docs)

        for i in range(0, self.n_iter):
            self.__gibbs_sampling(docs)
            print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", self.__perplexity(docs))

        topics = []
        n_topic_words_disp = 10
        for z in range(0, self.k):
            ids = self.nzw[z, :].argsort()
            topic_words = []
            for j in ids:
                topic_words.insert(0, vocab[j])
            topics.append(topic_words[0: min(n_topic_words_disp, len(topic_words))])

        for t in topics:
            print(t)

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

    def __init_with_data(self, docs):
        self.__init_A(docs)
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

    def __init_A(self, docs):
        n_docs = len(docs)
        self.p_co = np.zeros((self.n_words, self.n_words), np.float32)
        word_docs = [set() for _ in range(self.n_words)]
        for i, doc in enumerate(docs):
            words = set(doc)
            for w in words:
                word_docs[w].add(i)

        self.A = np.zeros((self.n_words, self.n_words), np.float32)
        lambda_v = np.zeros(self.n_words, np.float32)
        for w1 in range(self.n_words):
            docs1 = word_docs[w1]
            # print(docs1)
            lambda_v[w1] = np.log(n_docs / len(docs1))
            self.A[w1][w1] = len(docs1)
            for w2 in range(w1 + 1, self.n_words):
                if w1 == w2:
                    continue
                docs2 = word_docs[w2]
                si = docs1.intersection(docs2)
                su = docs1.union(docs2)
                self.A[w1][w2] = self.A[w2][w1] = len(si)
                self.p_co[w1][w2] = self.p_co[w2][w1] = len(si) / len(su)

        for j in range(self.n_words):
            for i in range(self.n_words):
                self.A[i][j] *= lambda_v[i]
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
    cv = textvectorizer.CountVectorizer(QUORA_DF_FILE, 50, 10000, remove_stopwords=True, docs=docs_words)
    print(len(cv.vocab), 'words in vocab')

    docs = list()
    for words in docs_words:
        doc = list()
        for w in words:
            widx = cv.word_dict.get(w, -1)
            if widx > -1:
                doc.append(widx)
        docs.append(doc)

    return docs, cv.word_dict, cv.vocab


urnlda = UrnLDA(k=5)
docs, word_idx_dict, vocab = process_quora()
urnlda.fit(docs, word_idx_dict, vocab)
