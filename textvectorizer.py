import pandas as pd
import numpy as np
from scipy import sparse
from collections import Counter


def gen_df(docs_file, dst_file):
    f = open(docs_file, encoding='utf-8')
    df_dict = dict()
    for line in f:
        words = line.strip().split(' ')
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

        self.word_idf_dict = {w: np.log(float(n_docs) / cnt) for w, cnt in df.itertuples(False, None)}

    def get_vec(self, text: str):
        words = text.strip().split(' ')
