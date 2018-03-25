import pandas as pd


def gen_df(docs_file, dst_file):
    f = open(docs_file, encoding='utf-8')
    df_dict = dict()
    for line in f:
        words = line.strip().split(' ')
        for w in words:
            cnt = df_dict.get(w, 0)
            df_dict[w] = cnt + 1
    f.close()
