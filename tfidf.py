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

    tups = [(w, cnt) for w, cnt in df_dict.items()]
    tups.sort(key=lambda x: -x[1])
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(tups, columns=['word', 'cnt']).to_csv(fout, index=False)


class TfIdf:
    def __init__(self, df_file, min_df, max_df):
        df = pd.read_csv(df_file)
        print(df.head())
