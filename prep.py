import re
import os
import pandas as pd
from config import *


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


src_doc_file = os.path.join(DATADIR, 'bizmsg.csv')
doc_file = os.path.join(DATADIR, 'docs-14k.csv')
# title_file = os.path.join(DATADIR, 'docs-14k-titles.csv')
content_file = os.path.join(DATADIR, 'docs-14k-content.csv')

# __fix_src_data()
__gen_sep_content_file(doc_file, content_file)
