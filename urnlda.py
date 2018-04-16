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

    def fit(self, docs, word_idx_dict, vocab):
        self.n_docs = len(docs)
        self.n_words = len(vocab)
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
                self.nzw[z, w] -= 1
                self.nz[z] -= 1

                pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                z = np.random.multinomial(1, pz / pz.sum()).argmax()
                self.Z[d][pos] = z

                self.ndz[d, z] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1

    def __init_with_data(self, docs):
        for d, doc in enumerate(docs):
            z_curdoc = []
            for w in doc:
                pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                z_tmp = np.random.multinomial(1, pz / pz.sum()).argmax()
                z_curdoc.append(z_tmp)
                self.ndz[d, z_tmp] += 1
                self.nzw[z_tmp, w] += 1
                self.nz[z_tmp] += 1
            self.Z.append(z_curdoc)

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
    cv = textvectorizer.CountVectorizer(QUORA_DF_FILE, 50, 10000, True)
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]

    docs = list()
    for content in contents:
        words = content.split(' ')
        doc = list()
        for w in words:
            widx = cv.word_dict.get(w, -1)
            if widx > -1:
                doc.append(widx)
        docs.append(doc)

    return docs, cv.word_dict, cv.vocab


urnlda = UrnLDA()
docs, word_idx_dict, vocab = process_quora()
urnlda.fit(docs, word_idx_dict, vocab)
