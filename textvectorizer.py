import pandas as pd
import numpy as np
from scipy import sparse
from collections import Counter


def gen_df(docs_file, dst_file):
    f = open(docs_file, encoding='utf-8')
    df_dict = dict()
    for line in f:
        words = set(line.strip().split(' '))
        for w in words:
            cnt = df_dict.get(w, 0)
            df_dict[w] = cnt + 1
    f.close()

    tups = [(w, cnt) for w, cnt in df_dict.items()]
    tups.sort(key=lambda x: -x[1])
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(tups, columns=['word', 'cnt']).to_csv(fout, index=False)


class TfIdf:
    def __init__(self, df_file, min_df, max_df, n_docs):
        df = pd.read_csv(df_file)
        df = df[df['cnt'].between(min_df, max_df)]

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
