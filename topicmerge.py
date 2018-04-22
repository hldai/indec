import os
import numpy as np
from scipy import sparse
from config import *
from topicmodel import TopicModel, topic_prob
import utils
import textvectorizer
from sklearn.metrics.pairwise import cosine_similarity


def reduce_topics(topics):
    pass


def __check_coherences():
    name = 'DC'
    # name = 'WP'
    # name = 'Austin'
    # name = 'Mark'

    vocab_file = os.path.join(QUORA_DATA_DIR, '{}_vocab.txt'.format(name))
    topic_file = os.path.join(QUORA_DATA_DIR, '{}_topics.txt'.format(name))
    tm = TopicModel(vocab_file, topic_file)
    print(len(tm.vocab), 'words in vocab')
    k = tm.n_topics

    word_idfs = textvectorizer.get_word_idfs(tm.vocab, QUORA_DF_FILE, QUORA_NUM_TOTAL_DOCS)

    contents = utils.load_docs_with_name(name, QUORA_ANSWER_TOK_LOWER_FILE, QUORA_NAME_DOC_FILE)
    D_codoc = utils.get_codoc_matrix(tm.vocab, contents)

    coherences = list()
    for i, t in enumerate(tm.topics):
        coherences.append(tm.coherence(t, D_codoc))
        print(i, coherences[-1])
        idxs = np.argpartition(-t, range(20))[:20]
        print(' '.join([tm.vocab[i] for i in idxs]))
    print()

    M_coh = 20
    cohs = list()
    coh_arr = np.zeros((k, k), np.float32)
    for i, t1 in enumerate(tm.topics):
        c = TopicModel.coherence(t1, D_codoc, M_coh)
        cohs.append(c)
        widxs1 = np.argpartition(-t1, range(M_coh))[:M_coh]
        for j in range(i + 1, k):
            t2 = tm.topics[j]
            widxs2 = np.argpartition(-t2, range(M_coh))[:M_coh]
            for w1 in widxs1:
                for w2 in widxs2:
                    coh_arr[i][j] += D_codoc[w1][w2] / D_codoc[w1][w1] / D_codoc[w2][w2]
            coh_arr[j][i] = coh_arr[i][j]
    print(cohs)
    for cs in coh_arr:
        print(' '.join(['{:06.3f}'.format(v) for v in cs]))

    print()
    t1, t2 = tm.topics[1], tm.topics[7]
    widxs1 = np.argpartition(-t1, range(M_coh))[:M_coh]
    widxs2 = np.argpartition(-t2, range(M_coh))[:M_coh]
    for w1 in widxs1:
        cohs = list()
        for w2 in widxs2:
            cohs.append(D_codoc[w1][w2] / D_codoc[w1][w1] / D_codoc[w2][w2])
        print(tm.vocab[w1], sum(cohs), word_idfs[w1])
        print(' '.join(['{:.3f}'.format(v) for v in cohs]))


def __check_doc_dists():
    name = 'DC'
    # name = 'WP'
    # name = 'Austin'
    # name = 'Mark'

    vocab_file = os.path.join(QUORA_DATA_DIR, '{}_vocab.txt'.format(name))
    topic_file = os.path.join(QUORA_DATA_DIR, '{}_topics.txt'.format(name))
    tm = TopicModel(vocab_file, topic_file)
    print(len(tm.vocab), 'words in vocab')

    word_idfs = textvectorizer.get_word_idfs(tm.vocab, QUORA_DF_FILE, QUORA_NUM_TOTAL_DOCS)

    contents = utils.load_docs_with_name(name, QUORA_ANSWER_TOK_LOWER_FILE, QUORA_NAME_DOC_FILE)
    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer(tm.vocab, remove_stopwords=True, words_exist=words_exist)
    X = cv.get_vecs(contents)

    topic_docs = [list() for _ in range(tm.n_topics)]
    for i, doc in enumerate(X):
        probs = tm.topic_probs(doc.indices, doc.data)
        topic_idx = np.argmax(probs)
        topic_docs[topic_idx].append(i)

    X_topics = sparse.lil_matrix((tm.n_topics, len(tm.vocab)))
    utils.disp_topics(tm.vocab, tm.topics, 20)
    for i, docs in enumerate(topic_docs):
        print('{}'.format(' '.join([str(d)for d in docs])))
        for d in docs:
            X_topics[i] += X[d]

    X_topics = textvectorizer.get_tfidf_vecs(X_topics, word_idfs)
    # topic_sims = np.zeros((tm.n_topics, tm.n_topics), np.float32)
    # for i, dt1 in enumerate(X_topics):
    #     for j in range(i + 1, tm.n_topics):
    #         dt2 = X_topics[j]
    topic_sims = cosine_similarity(X_topics)
    for ts in topic_sims:
        print(' '.join('{:.3f}'.format(v) for v in ts))


def __check_topic_match():
    # name = 'DC'
    name = 'WP'
    # name = 'Austin'
    # name = 'Mark'

    vocab_file = os.path.join(QUORA_DATA_DIR, '{}_vocab.txt'.format(name))
    topic_file = os.path.join(QUORA_DATA_DIR, '{}_topics.txt'.format(name))
    tm = TopicModel(vocab_file, topic_file)
    print(len(tm.vocab), 'words in vocab')

    for i, t in enumerate(tm.topics):
        idxs = np.argpartition(-t, range(20))[:20]
        print(' '.join([tm.vocab[i] for i in idxs]))

    contents = utils.load_docs_with_name(name, QUORA_ANSWER_TOK_LOWER_FILE, QUORA_NAME_DOC_FILE)
    docs_words = [content.split(' ') for content in contents]
    words_exist = utils.get_word_set(docs_words)
    cv = textvectorizer.CountVectorizer(tm.vocab, remove_stopwords=True, words_exist=words_exist)
    X = cv.get_vecs(contents)

    topic_docs = [list() for _ in range(tm.n_topics)]
    for i, doc in enumerate(X):
        probs = tm.topic_probs(doc.indices, doc.data)
        topic_idx = np.argmax(probs)
        topic_docs[topic_idx].append(i)

    meanp = np.log(1 / cv.n_words)

    match_arr = np.zeros((tm.n_topics, tm.n_topics), np.float32)
    for i, t1 in enumerate(tm.topics):
        for j in range(tm.n_topics):
            docs = topic_docs[j]
            probs = list()
            for d in docs:
                dv = X[d]
                probs.append(topic_prob(t1, dv.indices, dv.data) / np.sum(dv.data))
                # print(p)
                # prob += np.log(p)
            tmp = [p for p in probs if p > meanp]
            # print('{:.3f} '.format(prob), end='')
            print('{:03} '.format(len(tmp)), end='')
        print()


if __name__ == '__main__':
    __check_coherences()
    # __check_doc_dists()
    # __check_topic_match()
