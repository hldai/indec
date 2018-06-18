import re


def __seach_docs():
    # 专车 服务 刘备 公司 汽车 司机 出行 车 管理 夏侯渊
    words = ['演义', '三国', '15日', '京剧', '3月', '墓葬', '国家', '箭', '演员', '死后']
    f = open('d:/data/indec/docs-14k-minidocs-text-seg-new.txt', encoding='utf-8')
    for i, line in enumerate(f):
        cnt = 0
        occur_words = list()
        for w in words:
            if w in line:
                cnt += 1
                occur_words.append(w)
        # print(cnt)
        if cnt > 1:
            print(i, occur_words)
            print(line.replace(' ', ''))
    f.close()


__seach_docs()
