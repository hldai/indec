import pandas as pd
import re
import json
from utils import commonutils


def load_entity_name_to_minidoc_file(minidoc_info_file):
    name_doc_dict = dict()
    df = pd.read_csv(minidoc_info_file)
    for mdid, doc_id, entity_name in df.itertuples(False, None):
        mdids = name_doc_dict.get(entity_name, list())
        if not mdids:
            name_doc_dict[entity_name] = mdids
        mdids.append(mdid)
    return name_doc_dict


def load_entity_name_to_doc_file(filename):
    name_doc_dict = dict()
    f = open(filename, encoding='utf-8')
    for line in f:
        obj = json.loads(line)
        name_doc_dict[obj['entity_name']] = obj['docs']
    f.close()
    return name_doc_dict


def fix_src_data_mar(src_doc_file, dst_file):
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
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        df.to_csv(fout, index=False)


def fix_src_data_sep(src_doc_file, dst_file):
    f = open(src_doc_file, encoding='utf-8')
    next(f)
    data_tups = list()
    for i, line in enumerate(f):
        p = line.find(',')
        title, content = line[:p], line[p + 1:].strip()
        data_tups.append((title, content))
    f.close()

    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(data_tups, columns=['title', 'content']).to_csv(fout, index=False)
