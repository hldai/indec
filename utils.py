import json


def read_lines_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        return [line for line in f]


def load_entity_name_to_doc_file(filename):
    name_doc_dict = dict()
    f = open(filename, encoding='utf-8')
    for line in f:
        obj = json.loads(line)
        name_doc_dict[obj['entity_name']] = obj['docs']
    f.close()
    return name_doc_dict


def get_word_set(docs):
    words = set()
    for doc in docs:
        for w in doc:
            words.add(w)
    return words
