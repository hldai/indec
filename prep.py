import re


def __next_wiki_doc(fin):
    try:
        while True:
            line = next(fin)
            if line.startswith('<doc id=\"'):
                break
        m = re.match('<doc id=\"(\d+?)\" url=.*?title=\"(.*?)\">', line)
        assert m is not None

        doc_id, title = m.group(1), m.group(2)

        text = ''
        while True:
            line = next(fin)
            if line.startswith('</doc>'):
                break
            text += line
    except StopIteration:
        return None

    return doc_id, title, text


def __ambig_names_from_wiki(filename, dst_file):
    f = open(filename, encoding='utf-8')
    doc_cnt = 0
    name_target_used_cnts = dict()
    name_used_cnts = dict()
    while True:
        r = __next_wiki_doc(f)
        if r is None:
            break
        doc_id, title, text = r
        doc_cnt += 1

        miter = re.finditer('\[\[(.*?)\|(.*?)\]\]', text)
        for m in miter:
            s_to, s_from = m.group(1), m.group(2)
            if len(s_from) > 6 or len(s_to) > 10 or len(s_from) < 2:
                continue
            cnt = name_target_used_cnts.get((s_from, s_to), 0)
            name_target_used_cnts[(s_from, s_to)] = cnt + 1

            cnt = name_used_cnts.get(s_from, 0)
            name_used_cnts[s_from] = cnt + 1

        # if doc_cnt > 50000:
        #     break
        if doc_cnt % 100000 == 0:
            print(doc_cnt)
    f.close()

    name_target_used_cnt_tups = list(name_target_used_cnts.items())
    name_target_used_cnt_tups.sort(key=lambda x: -x[1])
    name_target_used_cnt_tups.sort(key=lambda x: x[0][0])

    fout = open(dst_file, 'w', encoding='utf-8')
    cur_name = ''
    cur_tups = list()
    i = 0
    while i < len(name_target_used_cnt_tups):
        (name, target), cnt = name_target_used_cnt_tups[i]
        if name != cur_name or i == len(name_target_used_cnt_tups) - 1:
            eligible = True

            n_used = name_used_cnts.get(cur_name, 0)
            if n_used < 50 or len(cur_tups) < 2 or (cur_name.endswith('年') and cur_name.startswith('2')):
                eligible = False

            if not cur_name.isupper():
                eligible = False
            # if len(cur_name) < 1 or not cur_name[0] in {'李', '王', '张', '刘', '陈', '杨', '赵'}:
            #     eligible = False

            if eligible:
                for (name, target), cnt in cur_tups:
                    if cnt / n_used > 0.7 or (cnt / n_used and len(cur_tups) > 10):
                        # print(name, target)
                        eligible = False
                        break

            if eligible:
                for (name, target), cnt in cur_tups:
                    fout.write('{}\t{}\t{}\t{:.2f}\n'.format(name, target, cnt, cnt / n_used))

            cur_tups = list()
        cur_name = name
        cur_tups.append(((name, target), cnt))
        i += 1
    fout.close()


wiki_file = 'd:/data/res/wiki-text-simplified.txt'
ambig_names_file = 'd:/data/indec/ambig-names-from-wiki.txt'
__ambig_names_from_wiki(wiki_file, ambig_names_file)
