import numpy as np


class Softmax:
    def predict(self, x):
        output = np.exp(x) / np.sum(np.exp(x))
        return output

    def loss(self, X, y):
        probs = self.predict(X)
        log_probs = -np.log(probs[y])

        return log_probs

    def diff_scores(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0

        return probs