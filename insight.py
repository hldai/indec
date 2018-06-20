import re


def __seach_docs():
    # words = ['钓鱼', '木', '故事', '运动', '协会', '消费者', '故里', '精彩', '文化', '老师']
    words = ['孙权']
    f = open('d:/data/indec/docs-14k-minidocs-text-seg-new.txt', encoding='utf-8')
    for i, line in enumerate(f):
        occur_words = list()
        for w in words:
            if w in line:
                occur_words.append(w)
        # print(cnt)
        # if len(occur_words) > 1:
        if len(occur_words) > 0:
            text = line.replace(' ', '')
            # if '姜太公' not in text:
            #     continue
            print(i, occur_words)
            print(text)
    f.close()


__seach_docs()
