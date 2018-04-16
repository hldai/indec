import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from config import *
import utils
import textvectorizer


def __process_quora():
    cv = textvectorizer.CountVectorizer(QUORA_DF_FILE, 50, 10000, True)
    print(cv.n_words, 'words in vocabulary')

    name = 'DC'
    all_doc_contents = utils.read_lines_to_list(QUORA_ANSWER_TOK_LOWER_FILE)
    name_doc_dict = utils.load_entity_name_to_doc_file(QUORA_NAME_DOC_FILE)
    doc_idxs = name_doc_dict[name]
    contents = [all_doc_contents[idx] for idx in doc_idxs]
    X = cv.get_vecs(contents)

    k = 10
    lda = LatentDirichletAllocation(k, learning_method='batch', doc_topic_prior=.1, topic_word_prior=0.01)
    X_new = lda.fit_transform(X)
    # for t in lda.components_:
    #     max_word_idxs = np.argpartition(-t, np.arange(10))[:10]
    #     for idx in max_word_idxs:
    #         print(cv.vocab[idx], end=' ')
    #     print()

    topic_cnts = {i: 0 for i in range(k)}
    for i, x in enumerate(X_new):
        max_topic_idxs = np.argpartition(-x, np.arange(3))[:3]
        topic_cnts[max_topic_idxs[0]] += 1
        # print(i + 1)
        # for tidx in max_topic_idxs:
        #     topic_dist = lda.components_[tidx]
        #     max_word_idxs = np.argpartition(-topic_dist, np.arange(10))[:10]
        #     topic_words = [cv.vocab[idx] for idx in max_word_idxs]
        #     print(x[tidx], ' '.join(topic_words))
        # print()
        # if i == 50:
        #     break
    for tidx, cnt in topic_cnts.items():
        print(tidx, cnt)
        max_word_idxs = np.argpartition(-lda.components_[tidx], np.arange(10))[:10]
        for idx in max_word_idxs:
            print('{}*{:.3f}'.format(cv.vocab[idx], lda.components_[tidx][idx]), end=' ')
        print()


__process_quora()
