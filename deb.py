# from numpy.ctypeslib import ndpointer
import ctypes
import pandas as pd
import numpy as np

df = pd.read_csv('d:/data/indec/docs-14k-minidocs-info-nodup.txt')
df = df.drop('mdid', axis=1)
print(df.head())
df['mdid'] = np.arange(df.shape[0])
print(df.head())
with open('d:/data/indec/docs-14k-minidocs-info-nodup.csv', 'w', encoding='utf-8', newline='\n') as fout:
    df.to_csv(fout, index=False, columns=['mdid', 'doc_id', 'entity_name'])
