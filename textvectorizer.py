import pandas as pd
import numpy as np
from scipy import sparse
from collections import Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


SPECIAL_WORDS = frozenset([
    '-lsb-', '-rsb-', '*', ';', '#', '·', '$', '`', '>', '%', '\\', '/', '--', '&'])


def get_word_idfs(vocab, df_file, n_docs):
    df = pd.read_csv(df_file)
    # TODO


def gen_df(docs_file, dst_file, to_lower=False, rm_one_time_words=True):
    f = open(docs_file, encoding='utf-8')
    df_dict = dict()
    for line in f:
        if to_lower:
            line = line.lower()
        words = set(line.strip().split(' '))
        for w in words:
            cnt = df_dict.get(w, 0)
            df_dict[w] = cnt + 1
    f.close()

    tups = [(w, cnt) for w, cnt in df_dict.items()]
    tups.sort(key=lambda x: -x[1])
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        df = pd.DataFrame(tups, columns=['word', 'cnt'])
        if rm_one_time_words:
            df = df[df['cnt'] > 1]
        df.to_csv(fout, index=False)


class CountVectorizer:
    def __init__(self, vocab_arg, remove_stopwords=False, remove_special_words=True, words_exist=None):
        self.vocab = list()
        self.word_cnts = dict()

        if isinstance(vocab_arg, tuple):
            df_file, min_df, max_df = vocab_arg
            df = pd.read_csv(df_file)
            df = df[df['cnt'].between(min_df, max_df)]
            for w, cnt in df.itertuples(False, None):
                if remove_stopwords and w in ENGLISH_STOP_WORDS:
                    continue
                if remove_special_words and w in SPECIAL_WORDS:
                    continue
                if words_exist is not None and w not in words_exist:
                    continue
                self.word_cnts[w] = cnt
                self.vocab.append(w)
        else:
            self.vocab = vocab_arg

        self.word_dict = {w: i for i, w in enumerate(self.vocab)}
        self.n_words = len(self.vocab)

    def get_vec(self, text: str):
        words = text.strip().split(' ')
        words_counter = Counter(words)
        data, rows, cols = list(), list(), list()
        for w, cnt in words_counter.items():
            word_idx = self.word_dict.get(w, None)
            if word_idx is None:
                continue
            rows.append(0)
            cols.append(word_idx)
            data.append(cnt)
        return sparse.csr_matrix((data, (rows, cols)), (1, self.n_words))

    def get_vecs(self, text_list, normalize=False):
        data, rows, cols = list(), list(), list()
        for i, text in enumerate(text_list):
            words = text.strip().split(' ')
            n_words = len(words)
            words_counter = Counter(words)
            for w, cnt in words_counter.items():
                word_idx = self.word_dict.get(w, None)
                if word_idx is None:
                    continue
                rows.append(i)
                cols.append(word_idx)
                if normalize:
                    data.append(cnt / n_words)
                else:
                    data.append(cnt)
        return sparse.csr_matrix((data, (rows, cols)), (len(text_list), self.n_words))


class TfIdf:
    def __init__(self, df_file, min_df, max_df, n_docs, remove_stopwords=False):
        df = pd.read_csv(df_file)
        df = df[df['cnt'].between(min_df, max_df)]

        if remove_stopwords:
            word_cnt = 0
            self.word_dict = dict()
            for w, cnt in df.itertuples(False, None):
                if w not in ENGLISH_STOP_WORDS:
                    self.word_dict[w] = (word_cnt, np.log(float(n_docs) / cnt))
                    word_cnt += 1
        else:
            self.word_dict = {w: (i, np.log(float(n_docs) / cnt)) for i, (w, cnt) in enumerate(
                df.itertuples(False, None))}
        self.n_words = len(self.word_dict)
        # self.word_idf_dict = {w: np.log(float(n_docs) / cnt) for w, cnt in df.itertuples(False, None)}

    def get_vec(self, text: str):
        words = text.strip().split(' ')
        words_counter = Counter(words)
        data, rows, cols = list(), list(), list()
        n_words_doc = 0
        for w, cnt in words_counter.items():
            tmp = self.word_dict.get(w, None)
            if tmp is None:
                continue
            n_words_doc += cnt

        for w, cnt in words_counter.items():
            tmp = self.word_dict.get(w, None)
            if tmp is None:
                continue
            idx, idf_val = tmp
            rows.append(0)
            cols.append(idx)
            data.append(float(cnt) / n_words_doc * idf_val)

        return sparse.csr_matrix((data, (rows, cols)), (1, self.n_words))

    def get_vecs(self, text_list):
        data, rows, cols = list(), list(), list()
        for i, text in enumerate(text_list):
            words = text.strip().split(' ')
            words_counter = Counter(words)
            n_words_doc = 0
            for w, cnt in words_counter.items():
                tmp = self.word_dict.get(w, None)
                if tmp is None:
                    continue
                n_words_doc += cnt

            for w, cnt in words_counter.items():
                tmp = self.word_dict.get(w, None)
                if tmp is None:
                    continue
                idx, idf_val = tmp
                rows.append(i)
                cols.append(idx)
                data.append(float(cnt) / n_words_doc * idf_val)

        return sparse.csr_matrix((data, (rows, cols)), (len(text_list), self.n_words))
