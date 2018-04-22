import numpy as np
import pandas as pd
import utils


def topic_prob(topic, word_idxs, cnts):
    prob = 0
    pw = topic / np.sum(topic)
    # print(pw[:10])
    for idx, cnt in zip(word_idxs, cnts):
        prob += np.log(pw[idx]) * cnt
    return prob


class TopicModel:
    def __init__(self, vocab_file, topic_file):
        self.vocab = utils.read_lines_to_list(vocab_file)
        df = pd.read_csv(topic_file, header=None)
        self.topics = df.as_matrix()
        self.n_topics = len(self.topics)
        # self.topics = np.zeros()

    def topic_probs(self, word_idxs, cnts):
        probs = np.zeros(self.n_topics, np.float32)
        for i, t in enumerate(self.topics):
            pw = t / np.sum(t)
            # print(pw[:10])
            for idx, cnt in zip(word_idxs, cnts):
                probs[i] += np.log(pw[idx]) * cnt
        return probs

    @staticmethod
    def coherence(t, D, M=10):
        word_idxs = np.argpartition(-t, range(M))[:M]
        s = 0
        for m in range(1, M):
            for l in range(m):
                vm, vl = word_idxs[m], word_idxs[l]
                s += np.log((D[vm][vl] + 1) / D[vl][vl])
        return s
