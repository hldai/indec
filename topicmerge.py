import os
import numpy as np
from config import *
from topicmodel import TopicModel


def merge_topics(topics):
    pass


if __name__ == '__main__':
    vocab_file = os.path.join(QUORA_DATA_DIR, 'wp_vocab.txt')
    topic_file = os.path.join(QUORA_DATA_DIR, 'wp_topics.txt')
    tm = TopicModel(vocab_file, topic_file)
    print(len(tm.vocab))
    for t in tm.topics:
        idxs = np.argpartition(-t, range(10))[:10]
        print(' '.join([tm.vocab[i] for i in idxs]))
