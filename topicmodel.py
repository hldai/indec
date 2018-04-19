import numpy as np
import pandas as pd
import utils


class TopicModel:
    def __init__(self, vocab_file, topic_file):
        self.vocab = utils.read_lines_to_list(vocab_file)
        df = pd.read_csv(topic_file, sep=' ', header=None)
        self.topics = df.as_matrix()
        # self.topics = np.zeros()

    def coherence(self, t_idx, D, M=10):
        t = self.topics[t_idx]
        word_idxs = np.argpartition(-t, range(M))[:M]
        s = 0
        for m in range(1, M):
            for l in range(m):
                vm, vl = word_idxs[m], word_idxs[l]
                s += np.log((D[vm][vl] + 1) / D[vl][vl])
        return s
