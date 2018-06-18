import re
import os
import json
import pandas as pd
from config import *
import textvectorizer
import utils
from sklearn.metrics.pairwise import cosine_similarity


SENT_SEG_CHARS = '。.？?！!'
SENT_SEG_CONTINUE_CHARS = '。.？?！!”'

SUB_SENT_SEG_CHARS = '，？。.,、‘’“”、'


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


def __gen_minidocs_with_specific_name():
    df = pd.read_csv(WC_ENTITY_NAMES_FILE, header=None)
    name_doc_dict = utils.load_entity_name_to_minidoc_file(WC_MINIDOC_INFO_NODUP_FILE)
    for ch_name, en_name in df.itertuples(False, None):
        if en_name != 'swk':
            continue

        all_doc_contents = utils.read_lines_to_list(WC_MINIDOC_TEXT_NODUP_FILE)
        doc_idxs = name_doc_dict[ch_name]
        contents = [all_doc_contents[idx] for idx in doc_idxs]
        print(len(contents), 'docs')
        fout = open('d:/data/indec/entity-data/{}-mini.txt'.format(en_name), 'w', encoding='utf-8', newline='\n')
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


def split_sentences(text):
    sents = list()
    p, p_prev = 0, 0
    l = len(text)
    while p < l:
        # print(p)
        ch = text[p]
        if ch in SENT_SEG_CHARS:
            while True:
                p += 1
                if p >= l:
                    break
                ch = text[p]
                if ch not in SENT_SEG_CONTINUE_CHARS:
                    break
            sent = text[p_prev:p].strip()
            if sent:
                sents.append(sent)
            p_prev = p
        else:
            p += 1
    sent = text[p_prev:].strip()
    if sent:
        sents.append(sent)
    return sents


def __sent_split():
    f = open(content_file, encoding='utf-8')
    fout = open(WC_SENT_FILE, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        sents = split_sentences(line.strip())
        for sent in sents:
            fout.write('{}\n'.format(sent))
        fout.write('\n')
        # if i == 5:
        #     break
    f.close()
    fout.close()


def __read_doc_sents(fin):
    sents = list()
    try:
        for line in fin:
            line = line.strip()
            if not line:
                return sents
            sents.append(line)
    except StopIteration:
        return None


def __gen_minidocs():
    entity_names = utils.read_lines_to_list(entity_names_file)
    f = open(WC_SENT_FILE, encoding='utf-8')
    fout_text = open(WC_MINIDOC_TEXT_FILE, 'w', encoding='utf-8', newline='\n')
    n_context_sents = 2
    doc_cnt, minidoc_cnt = 0, 0
    minidocs_info_list = list()
    while True:
        doc_sents = __read_doc_sents(f)
        if doc_sents is None:
            break
        for name in entity_names:
            i = 0
            while i < len(doc_sents):
                sent = doc_sents[i]
                if name not in sent:
                    i += 1
                    continue
                s_idx_beg = max(i - n_context_sents, 0)
                p = i + 1
                max_hit_pos = i
                while p < len(doc_sents):
                    if name in doc_sents[p]:
                        max_hit_pos = p
                    if p - max_hit_pos >= n_context_sents * 2:
                        break
                    p += 1
                i = p + 1
                s_idx_end = min(max_hit_pos + n_context_sents + 1, len(doc_sents))

                minidoc_text = ''.join(doc_sents[s_idx_beg:s_idx_end])
                minidocs_info_list.append((minidoc_cnt, doc_cnt, name))
                # minidoc = {'mdid': minidoc_cnt, 'doc_id': doc_cnt, 'text': minidoc_text, 'entity_name': name}
                minidoc_cnt += 1
                # fout.write('{}\n'.format(json.dumps(minidoc, ensure_ascii=False)))
                fout_text.write('{}\n'.format(minidoc_text))
                # print(name, cnt)
                # for s in doc_sents[s_idx_beg:s_idx_end]:
                #     print(s)
                # print(doc_sents[s_idx_beg:s_idx_end])
                # print()
        doc_cnt += 1
        # if doc_cnt > 10:
        #     break
        if doc_cnt % 1000 == 0:
            print(doc_cnt)
    f.close()
    # fout.close()
    fout_text.close()
    print(doc_cnt, 'docs,', minidoc_cnt, 'minidocs')
    df = pd.DataFrame(minidocs_info_list, columns=['mdid', 'doc_id', 'entity_name'])
    with open(WC_MINIDOC_INFO_FILE, 'w', encoding='utf-8', newline='\n') as fout:
        df.to_csv(fout, index=False)


def __filter_duplicate_minidocs():
    df_minidocs = pd.read_csv(WC_MINIDOC_INFO_FILE)
    # print(df_minidocs.head())
    all_doc_contents = utils.read_lines_to_list(WC_MINIDOC_TEXT_SEG_FILE)
    cv = textvectorizer.CountVectorizer((WC_DF_FILE, 100, 2000), remove_stopwords=True)
    print(cv.n_words, 'words in vocab')
    X = cv.get_vecs(all_doc_contents)
    n_docs = len(all_doc_contents)
    print(n_docs, 'docs', X.shape)
    dup_docs = set()
    for i, x1 in enumerate(X):
        cur_name = df_minidocs['entity_name'][i]
        # print(cur_name)
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
            if sim > 0.9 and cur_name == df_minidocs['entity_name'][j]:
                # print(i, j, minidocs[i]['entity_name'], minidocs[j]['entity_name'])
                dup_docs.add(j)

        # if i == 3:
        #     break

    # exit()
    dup_docs_list = list(dup_docs)
    dup_docs_list.sort()
    print(dup_docs_list[:30])

    # TODO mdid not correct
    df_fil = df_minidocs.drop(dup_docs_list)
    with open(WC_MINIDOC_INFO_NODUP_FILE, 'w', encoding='utf-8', newline='\n') as fout:
        df_fil.to_csv(fout, index=False)

    utils.remove_lines(WC_MINIDOC_TEXT_FILE, dup_docs, WC_MINIDOC_TEXT_NODUP_FILE)
    utils.remove_lines(WC_MINIDOC_TEXT_SEG_FILE, dup_docs, WC_MINIDOC_TEXT_SEG_NODUP_FILE)


def __sent_visited_tfidf(sent_tfidf, vst_sent_tfidf_list):
    for v in vst_sent_tfidf_list:
        sim = cosine_similarity(sent_tfidf, v)
        if sim > 0.9:
            return True
    return False


def __sent_visited(sent, vst_sents):
    for s in vst_sents:
        if s == sent:
            return True
    return False


def __find_duplicate(cur_text, texts):
    sub_sents = re.split('[{}]+'.format(SUB_SENT_SEG_CHARS), cur_text)

    tmp = list()
    for s in sub_sents:
        s = s.strip()
        if s:
            tmp.append(s)
    sub_sents = tmp

    if len(sub_sents) == 1:
        for i, text in enumerate(texts):
            if cur_text == text:
                return i

    min_match_num = 2
    for i, text in enumerate(texts):
        match_cnt = 0
        for s in sub_sents:
            if len(s) < 5:
                continue
            p = text.find(s)
            if p < 0:
                continue

            if (p == 0 or text[p - 1] in SUB_SENT_SEG_CHARS) and (
                    p + len(s) >= len(text) or text[p + len(s)] in SUB_SENT_SEG_CHARS):
                match_cnt += 1
                # print(s)
                if match_cnt == min_match_num:
                    break
            # print(cur_text)
            # print(i, text)
            # exit()
        if match_cnt == min_match_num:
            return i

    return -1


def __minidocs_for_name(entity_name, fout_text, fout_seg_text):
    print(entity_name)
    n_context_sents = 2
    # cv = textvectorizer.CountVectorizer((WC_DF_FILE, 100, 2000), remove_stopwords=True)
    # name_sent_vecs = list()
    name_sents = list()

    doc_cnt, minidoc_cnt = 0, 0
    minidocs_info_list = list()
    minidoc_texts, minidoc_seg_texts = list(), list()
    f = open(WC_SENT_FILE, encoding='utf-8')
    f_seg = open(WC_SEG_SENT_FILE, encoding='utf-8')
    while True:
        doc_sents = __read_doc_sents(f)
        doc_seg_sents = __read_doc_sents(f_seg)
        if doc_sents is None:
            break
        # print(doc_sents)
        i = 0
        while i < len(doc_sents):
            sent = doc_sents[i]
            if entity_name not in sent:
                i += 1
                continue

            # seg_sent = doc_seg_sents[i]
            # sent_tfidf = cv.get_vec(seg_sent)

            if __sent_visited(sent, name_sents):
                # print(sent)
                i += 1
                continue

            # name_sent_vecs.append(cv.get_vec(seg_sent))
            name_sents.append(sent)

            s_idx_beg = max(i - n_context_sents, 0)
            p = i + 1
            max_hit_pos = i
            while p < len(doc_sents):
                sent_tmp = doc_sents[p]
                if entity_name in sent_tmp:
                    # seg_sent_tmp = doc_seg_sents[p]
                    # sent_tfidf_tmp = cv.get_vec(seg
                    # _sent_tmp)
                    if not __sent_visited(sent_tmp, name_sents):
                        max_hit_pos = p
                        # name_sent_vecs.append(sent_tfidf_tmp)
                        name_sents.append(sent_tmp)
                if p - max_hit_pos >= n_context_sents * 2:
                    break
                p += 1
            i = p + 1
            s_idx_end = min(max_hit_pos + n_context_sents + 1, len(doc_sents))

            minidoc_text = ''.join(doc_sents[s_idx_beg:s_idx_end])
            dup_idx = __find_duplicate(minidoc_text, minidoc_texts)
            if dup_idx > -1:
                # print(minidoc_text)
                # print(minidoc_texts[dup_idx])
                # print()
                if len(minidoc_text) > len(minidoc_texts[dup_idx]):
                    minidoc_texts[dup_idx] = minidoc_text
                    minidocs_info_list[dup_idx] = (doc_cnt, entity_name)
                    minidoc_seg_texts[dup_idx] = ' '.join(doc_seg_sents[s_idx_beg:s_idx_end])
                continue

            # print(minidoc_text)
            minidoc_texts.append(minidoc_text)
            minidocs_info_list.append((doc_cnt, entity_name))
            minidoc_seg_text = ' '.join(doc_seg_sents[s_idx_beg:s_idx_end])
            minidoc_seg_texts.append(minidoc_seg_text)
            minidoc_cnt += 1
        doc_cnt += 1
        # print(doc_cnt)
        # if minidoc_cnt >= 100:
        #     break
        if doc_cnt % 1000 == 0:
            print(doc_cnt)
    f.close()
    f_seg.close()

    for t in minidoc_texts:
        fout_text.write('{}\n'.format(t))
    for t in minidoc_seg_texts:
        fout_seg_text.write('{}\n'.format(t))

    return minidocs_info_list


def __gen_minidocs_new():
    entity_names = utils.read_lines_to_list(entity_names_file)

    minidocs_info_list = list()
    fout_text = open('d:/data/indec/docs-14k-minidocs-text-new.txt', 'w', encoding='utf-8', newline='\n')
    fout_seg_text = open('d:/data/indec/docs-14k-minidocs-text-seg-new.txt', 'w', encoding='utf-8', newline='\n')
    for i, entity_name in enumerate(entity_names):
        info_list_tmp = __minidocs_for_name(entity_name, fout_text, fout_seg_text)
        minidocs_info_list += info_list_tmp
        if i == 1:
            break

    fout_text.close()
    fout_seg_text.close()
    minidocs_info_list = [(i, doc_id, name) for i, (doc_id, name) in enumerate(minidocs_info_list)]
    with open('d:/data/indec/docs-14k-minidocs-info-new.txt', 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(minidocs_info_list, columns=['mdid', 'doc_id', 'entity_name']).to_csv(fout, index=False)


src_doc_file = os.path.join(WC_DATADIR, 'bizmsg.csv')
doc_file = os.path.join(WC_DATADIR, 'docs-14k.csv')
# title_file = os.path.join(DATADIR, 'docs-14k-titles.csv')
content_file = os.path.join(WC_DATADIR, 'docs-14k-content.txt')
entity_names_file = os.path.join(WC_DATADIR, 'entities.txt')
# name_doc_file = os.path.join(DATADIR, 'name-doc.txt')

# __fix_src_data()
# __gen_sep_content_file(doc_file, content_file)
# __filter_duplicate_docs()
# textvectorizer.gen_df(WC_SEG_DOC_CONTENT_NODUP_FILE, WC_DF_FILE)
# __gen_name_to_doc_file(entity_names_file, WC_DOC_INFO_NODUP_FILE, WC_NAME_DOC_ND_FILE)
# __gen_docs_with_specific_name()
# __sent_split()

# __gen_minidocs()
# __filter_duplicate_minidocs()
# __gen_minidocs_with_specific_name()

__gen_minidocs_new()
