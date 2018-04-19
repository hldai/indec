import os
import numpy as np
from config import *
from topicmodel import TopicModel
import utils
import textvectorizer


def reduce_topics(topics):
    pass


def __check_coherences():
    # name = 'DC'
    name = 'WP'
    # name = 'Austin'
    # name = 'Mark'

    vocab_file = os.path.join(QUORA_DATA_DIR, 'wp_vocab.txt')
    topic_file = os.path.join(QUORA_DATA_DIR, 'wp_topics.txt')
    tm = TopicModel(vocab_file, topic_file)
    print(len(tm.vocab), 'words in vocab')

    contents = utils.load_docs_with_name(name, QUORA_ANSWER_TOK_LOWER_FILE, QUORA_NAME_DOC_FILE)
    D_codoc = utils.get_codoc_matrix(tm.vocab, contents)

    coherences = list()
    for i, t in enumerate(tm.topics):
        coherences.append(tm.coherence(t, D_codoc))
        print(i, coherences[-1])
        idxs = np.argpartition(-t, range(10))[:10]
        print(' '.join([tm.vocab[i] for i in idxs]))
    print()

    for i, t1 in enumerate(tm.topics):
        for j in range(i + 1, len(tm.topics)):
            t2 = tm.topics[j]
            newt = t1 + t2
            c = tm.coherence(t1 + t2, D_codoc)
            print(i, j, c, coherences[i] - c, coherences[j] - c, (coherences[i] + coherences[j]) / 2 - c)
            idxs = np.argpartition(-newt, range(10))[:10]
            print(' '.join([tm.vocab[i] for i in idxs]))


def __check_doc_dists():
    # name = 'DC'
    name = 'WP'
    # name = 'Austin'
    # name = 'Mark'

    vocab_file = os.path.join(QUORA_DATA_DIR, 'wp_vocab.txt')
    topic_file = os.path.join(QUORA_DATA_DIR, 'wp_topics.txt')
    tm = TopicModel(vocab_file, topic_file)
    print(len(tm.vocab), 'words in vocab')

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

    utils.disp_topics(tm.vocab, tm.topics)
    for docs in topic_docs:
        print(docs)


if __name__ == '__main__':
    # __check_coherences()
    __check_doc_dists()
