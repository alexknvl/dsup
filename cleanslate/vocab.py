import numpy as np
import scipy.sparse


class Vocabulary(object):
    __slots__ = ('words', 'index')

    def __getstate__(self):
        return (self.words, self.index)

    def __setstate__(self, state):
        self.words, self.index = state

    def __init__(self, words=None, index=None):
        self.words = words if words is not None else []
        if index is not None:
            self.index = index
        else:
            self.index = dict(map(reversed, enumerate(self.words)))

    def update(self, w):
        if w not in self.index:
            self.index[w] = len(self.words)
            self.words.append(w)
            return len(self.words) - 1
        return self.index[w]

    def get(self, w):
        return self.index.get(w, -1)

    def resolve(self, w, immutable=False):
        self.get(w) if immutable else self.update(w)

    def compress(self, predicate):
        new_words = []
        new_index = {}
        remapping = [0] * len(self.words)

        for i, w in enumerate(self.words):
            if predicate(i, w):
                new_index[w] = len(new_words)
                remapping[i] = len(new_words)
                new_words.append(w)
            else:
                remapping[i] = -1

        self.index = new_index
        self.words = new_words

        return remapping

    def as_matrix(self, wm):
        x = scipy.sparse.lil_matrix((len(wm), len(self.words)))
        for i, wv in enumerate(wm):
            for f in wv:
                j = self.index.get(f, -1)
                if j >= 0:
                    x[i, j] = 1
        return x.tocsr()

    def as_vector(self, wv):
        x = np.zeros(len(wv))
        for f in wv:
            j = self.index.get(f, -1)
            if j >= 0:
                x[j] = 1
        return x

    def __len__(self):
        return len(self.words)
