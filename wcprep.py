import re
import os
import json
import pandas as pd
from config import *
import textvectorizer
import utils
from sklearn.metrics.pairwise import cosine_similarity


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


def __gen_name_to_doc_file(entity_names_file, doc_file, dst_file):
    entity_names = pd.read_csv(entity_names_file, header=None).as_matrix().flatten()
    df = pd.read_csv(doc_file, na_filter=False)

    name_doc_dict = {name: list() for name in entity_names}
    for idx, content in enumerate(df['content']):
        for name in entity_names:
            if name in content:
                name_doc_dict[name].append(idx)

    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for name, docs in name_doc_dict.items():
        fout.write('{}\n'.format(json.dumps({'entity_name': name, 'docs': docs}, ensure_ascii=False)))
    fout.close()


def __gen_docs_with_specific_name():
    df = pd.read_csv(WC_ENTITY_NAMES_FILE, header=None)
    for ch_name, en_name in df.itertuples(False, None):
        if en_name != 'xhd':
            continue

        all_doc_contents = utils.read_lines_to_list(WC_DOC_CONTENT_NODUP_FILE)
        name_doc_dict = utils.load_entity_name_to_doc_file(WC_NAME_DOC_ND_FILE)
        doc_idxs = name_doc_dict[ch_name]
        contents = [all_doc_contents[idx] for idx in doc_idxs]
        print(len(contents), 'docs')
        fout = open('d:/data/indec/{}.txt'.format(en_name), 'w', encoding='utf-8', newline='\n')
        for text in contents:
            fout.write('{}\n'.format(text.strip()))
        fout.close()

        break


def __filter_duplicate_docs():
    all_doc_contents = utils.read_lines_to_list(WC_SEG_DOC_CONTENT_FILE)
    cv = textvectorizer.CountVectorizer((WC_DF_FILE, 100, 2000), remove_stopwords=True)
    print(cv.n_words, 'words in vocab')
    X = cv.get_vecs(all_doc_contents)
    n_docs = len(all_doc_contents)
    print(n_docs, 'docs', X.shape)
    dup_docs = set()
    for i, x1 in enumerate(X):
        if i % 100 == 0:
            print(i)
        # print(i)

        if i in dup_docs:
            continue

        for j in range(i + 1, n_docs):
            if j in dup_docs:
                continue
            sim = cosine_similarity(x1, X[j])
            # if 0.8 < sim < 0.9:
            #     print(i, j, sim)
            if sim > 0.8:
                dup_docs.add(j)

        # if i == 5:
        #     break

    # exit()
    doc_info_df = pd.read_csv(doc_file)
    dup_docs_list = list(dup_docs)
    dup_docs_list.sort()
    print(dup_docs_list[:30])
    df_fil = doc_info_df.drop(dup_docs_list)
    with open(WC_DOC_INFO_NODUP_FILE, 'w', encoding='utf-8', newline='\n') as fout:
        df_fil.to_csv(fout, index=False)

    utils.remove_lines(WC_DOC_CONTENT_FILE, dup_docs, WC_DOC_CONTENT_NODUP_FILE)
    utils.remove_lines(WC_SEG_DOC_CONTENT_FILE, dup_docs, WC_SEG_DOC_CONTENT_NODUP_FILE)


src_doc_file = os.path.join(WC_DATADIR, 'bizmsg.csv')
doc_file = os.path.join(WC_DATADIR, 'docs-14k.csv')
# title_file = os.path.join(DATADIR, 'docs-14k-titles.csv')
content_file = os.path.join(WC_DATADIR, 'docs-14k-content.csv')
entity_names_file = os.path.join(WC_DATADIR, 'entities.txt')
# name_doc_file = os.path.join(DATADIR, 'name-doc.txt')

# __fix_src_data()
# __gen_sep_content_file(doc_file, content_file)
# __filter_duplicate_docs()
# textvectorizer.gen_df(WC_SEG_DOC_CONTENT_NODUP_FILE, WC_DF_FILE)
# __gen_name_to_doc_file(entity_names_file, WC_DOC_INFO_NODUP_FILE, WC_NAME_DOC_ND_FILE)
__gen_docs_with_specific_name()
