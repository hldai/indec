import json
import textvectorizer
from config import *


def __load_entity_name_to_doc():
    name_doc_dict = dict()
    f = open(NAME_DOC_FILE, encoding='utf-8')
    for line in f:
        obj = json.loads(line)
        name_doc_dict[obj['entity_name']] = obj['docs']
    f.close()
    return name_doc_dict


def __get_dists(doc_idxs):
    print(doc_idxs[:10])


def __entity_disamb():
    tv = textvectorizer.TfIdf(DF_FILE, 5, 62000, 14357)

    name_doc_dict = __load_entity_name_to_doc()
    __get_dists(name_doc_dict['曹操'])


__entity_disamb()
