from config import *
from collections import namedtuple
import textvectorizer
import re
from bs4 import BeautifulSoup
import pandas as pd
import json


QuoraQA = namedtuple('QuoraQA', ['id', 'question', 'answer_score', 'answer_text', 'username',
                                 'twitter_id'])


def __next_quora_data_block(fin):
    try:
        qaid = int(next(fin).strip())
        question = next(fin).strip()
        answer_score = next(fin).strip()
        answer_text = next(fin).strip()
        answer_text = answer_text[14:]
        if len(answer_text) > 0 and answer_text[-1] == '"':
            answer_text = answer_text[:-1]
        answer_text = re.sub('\\\\"', '"', answer_text)
        # m = re.search(r'\\"', answer_text)
        # if m:
        #     print(m.group())
        username = next(fin).strip()
        twitter_id = next(fin).strip()
    except StopIteration:
        return None
    return QuoraQA(qaid, question, answer_score, answer_text, username, twitter_id)


def __gen_answer_text_file():
    f = open(quora_user_qa_file, encoding='utf-8', errors='ignore')
    fout = open(answer_text_file, 'w', encoding='utf-8', newline='\n')
    for line in f:
        qa = json.loads(line)
        fout.write('{}\n'.format(qa['answer']))
    f.close()
    fout.close()


def __check_ner_result():
    name_dict = dict()
    f = open(ner_result_file, encoding='utf-8')
    for line in f:
        vals = line.strip().split('\t')
        cnt = name_dict.get(vals[4], 0)
        name_dict[vals[4]] = cnt + 1
    f.close()

    tups = list(name_dict.items())
    tups.sort(key=lambda x: -x[1])
    with open(ner_result_stat_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(tups, columns=['name', 'cnt']).to_csv(fout, index=False)
    # for k, v in tups:
    #     print(k, v)


def __get_qa_pair_from_file(filename):
    with open(filename, encoding='utf-8', errors='ignore') as f:
        q = next(f).strip()
        a = next(f).strip()
    return q, a


def __merge_qa_pairs():
    fout = open(qa_pairs_file, 'w', encoding='utf-8', newline='\n')
    for i, filename in enumerate(os.listdir(qa_pair_dir)):
        file_path = os.path.join(qa_pair_dir, filename)
        q, a = __get_qa_pair_from_file(file_path)
        # print(q)
        # print(a)
        fout.write('{}\n{}\n'.format(q, a))
        if i % 10000 == 0:
            print(i)
    fout.close()


def __load_qa_pairs():
    f = open(qa_pairs_file, encoding='utf-8')
    qa_dict = dict()
    for line in f:
        q = re.sub('\\\\"', '"', line.strip())
        alist = qa_dict.get(q, list())
        if not alist:
            qa_dict[q] = alist
        answer = next(f).strip()
        alist.append(re.sub('\\\\"', '"', answer))
        # qa_dict[line.strip()] = next(f).strip()
    f.close()
    return qa_dict


def __get_user_answers_from_profile_page(filename):
    # with open(filename, encoding='utf-8', errors='ignore') as f:
    with open(filename, encoding='gbk', errors='ignore') as f:
        html_str = f.read()
    qa_pattern = '<span class="question_text">.*?rendered_qtext">(.*?)</span>.*?' \
                 '<div class="feed_item_answer.*?rendered_qtext">(.*?)</span>'
    miter = re.finditer(qa_pattern, html_str)
    user_qa_list = list()
    for m in miter:
        soup = BeautifulSoup(m.group(1), 'html.parser')
        for br in soup.find_all("br"):
            br.replace_with(" ")
        q = soup.text
        soup = BeautifulSoup(m.group(2), 'html.parser')
        for br in soup.find_all("br"):
            br.replace_with(" ")
        a = soup.text
        user_qa_list.append((q, a))
    return user_qa_list


def __get_user_answers_from_profile():
    cnt, cnt_found = 0, 0
    cnt_answer = 0
    qa_dict = __load_qa_pairs()
    fout = open(quora_user_qa_file, 'w', encoding='utf-8', newline='\n')
    for i, user_folder in enumerate(os.listdir(user_profile_dir)):
        user_answers_file = os.path.join(user_profile_dir, user_folder, 'answers')
        if not os.path.isfile(user_answers_file):
            continue
        # user_answers_file = 'd:/data/quora/quora-raw/profiles/A-Productions/answers'
        user_qa_list = __get_user_answers_from_profile_page(user_answers_file)
        cnt += len(user_qa_list)
        if i % 1000 == 0:
            print(i)
        for q, a in user_qa_list:
            answer_list = qa_dict.get(q, list())
            if not answer_list:
                continue
                # print(q)
            cnt_found += 1
            full_answer = None
            for cand_answer in answer_list:
                if cand_answer[:10] == a[:10]:
                    full_answer = cand_answer
                    cnt_answer += 1
                    break
            if full_answer is not None:
                user_qa_obj = {'username': user_folder, 'question': q, 'answer': full_answer}
                fout.write('{}\n'.format(json.dumps(user_qa_obj, ensure_ascii=False)))
            # if full_answer is None:
            #     print(a)
            #     for cand_answer in answer_list:
            #         print(cand_answer)
            #     print()
        # if i == 100:
        #     break
    print(cnt, cnt_found, cnt_answer)
    fout.close()


def __sort_questions():
    f = open(qa_pairs_file, encoding='utf-8')
    qlist = list()
    for line in f:
        next(f)
        qlist.append(line)
    f.close()

    fout = open(os.path.join(QUORA_DATA_DIR, 'tmp.txt'), 'w', encoding='utf-8', newline='\n')
    qlist.sort()
    for q in qlist:
        fout.write(q)
    fout.close()


def __answer_text_to_lower():
    f = open(QUORA_ANSWER_TOK_FILE, encoding='utf-8')
    fout = open(QUORA_ANSWER_TOK_LOWER_FILE, 'w', encoding='utf-8', newline='\n')
    for line in f:
        fout.write(line.lower())
    f.close()
    fout.close()


def __gen_name_to_doc_file():
    df = pd.read_csv(QUORA_NER_NAME_CNT_FILE)
    df = df[df['cnt'] > 2]
    entity_names = df.as_matrix(['name']).flatten()
    print(len(entity_names), 'names')

    name_doc_dict = {name: list() for name in entity_names}

    f = open(QUORA_DATA_FILE, encoding='utf-8')
    for i, line in enumerate(f):
        qa_obj = json.loads(line)
        for name in entity_names:
            if name in qa_obj['answer']:
                name_doc_dict[name].append(i)

        if (i + 1) % 10000 == 0:
            print(i + 1)

    fout = open(QUORA_NAME_DOC_FILE, 'w', encoding='utf-8', newline='\n')
    for name, docs in name_doc_dict.items():
        fout.write('{}\n'.format(json.dumps({'entity_name': name, 'docs': docs}, ensure_ascii=False)))
    fout.close()


ner_result_file = os.path.join(QUORA_DATA_DIR, 'answer-text-ner.txt')
answer_text_file = os.path.join(QUORA_DATA_DIR, 'answer-text.txt')
qa_pair_dir = os.path.join(QUORA_DATA_DIR, 'origin_data')
user_profile_dir = os.path.join(QUORA_DATA_DIR, 'quora-raw/profiles')
qa_pairs_file = os.path.join(QUORA_DATA_DIR, 'qa-pairs.txt')
quora_user_qa_file = os.path.join(QUORA_DATA_DIR, 'quora-user-qa.json')
ner_result_stat_file = os.path.join(QUORA_DATA_DIR, 'ner-name-cnts.txt')
# __merge_qa_pairs()
# __get_user_answers_from_profile()
# __gen_answer_text_file()
# __check_ner_result()
# __answer_text_to_lower()
# textvectorizer.gen_df(QUORA_ANSWER_TOK_FILE, QUORA_DF_FILE, True)
__gen_name_to_doc_file()
