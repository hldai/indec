import re
import os
import json
import pandas as pd
from config import *
import tfidf


def __fix_src_data():
    biz_ids, titles, contents = list(), list(), list()
    f = open(src_doc_file, encoding='utf-8')
    next(f)
    for i, line in enumerate(f):
        m = re.match('(\d+)_(\d+)_\d+,(.*?),(http://.*?|),(.*)', line)
        if m is None:
            # m = re.match('(\d+)_(\d+)_\d+,(.*?),http://.*?,(.*)', line)
            print(i)
            print(line)
        print(m.group(1), m.group(3))
        biz_ids.append(m.group(1).strip())
        titles.append(m.group(3).strip())
        contents.append(m.group(5).strip())
    f.close()

    df = pd.DataFrame({'biz_id': biz_ids, 'title': titles, 'content': contents})
    with open(doc_file, 'w', encoding='utf-8', newline='\n') as fout:
        df.to_csv(fout, index=False)


def __gen_sep_content_file(doc_file, dst_content_file):
    df = pd.read_csv(doc_file)
    with open(dst_content_file, 'w', encoding='utf-8', newline='\n') as fout:
        for text in df['content']:
            fout.write('{}\n'.format(text))


def __gen_name_to_doc_file():
    entity_names = pd.read_csv(entity_names_file, header=None).as_matrix().flatten()
    df = pd.read_csv(doc_file, na_filter=False)

    name_doc_dict = {name: list() for name in entity_names}
    for idx, content in enumerate(df['content']):
        for name in entity_names:
            if name in content:
                name_doc_dict[name].append(idx)

    fout = open(name_doc_file, 'w', encoding='utf-8', newline='\n')
    for name, docs in name_doc_dict.items():
        fout.write('{}\n'.format(json.dumps({'entity_name': name, 'docs': docs}, ensure_ascii=False)))
    fout.close()


src_doc_file = os.path.join(DATADIR, 'bizmsg.csv')
doc_file = os.path.join(DATADIR, 'docs-14k.csv')
# title_file = os.path.join(DATADIR, 'docs-14k-titles.csv')
content_file = os.path.join(DATADIR, 'docs-14k-content.csv')
seg_content_file = os.path.join(DATADIR, 'docs-14k-content-seg.txt')
df_file = os.path.join(DATADIR, 'docs-14k-words-df.txt')
entity_names_file = os.path.join(DATADIR, 'entities.txt')
name_doc_file = os.path.join(DATADIR, 'name-doc.txt')

# __fix_src_data()
# __gen_sep_content_file(doc_file, content_file)
# tfidf.gen_df(seg_content_file, df_file)
__gen_name_to_doc_file()