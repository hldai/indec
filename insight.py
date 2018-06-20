import re


SUB_SENT_SEG_CHARS = '，？。.,、‘’“”、'


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


def __seach_docs():
    # words = ['钓鱼', '木', '故事', '运动', '协会', '消费者', '故里', '精彩', '文化', '老师']
    words = ['自在', '神', '破坏', '超级', '形态', '合影', '集', '悟空', '状态', '情况']
    f = open('d:/data/indec/docs-14k-minidocs-text-seg-new.txt', encoding='utf-8')
    for i, line in enumerate(f):
        occur_words = list()
        for w in words:
            if w in line:
                occur_words.append(w)
        # print(cnt)
        # if len(occur_words) > 1:
        if len(occur_words) > 1:
            text = line.replace(' ', '')
            if '悟空' not in text:
                continue
            print(i, occur_words)
            print(text)
    f.close()


__seach_docs()

