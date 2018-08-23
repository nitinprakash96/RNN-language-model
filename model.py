import numpy as np
import sys
from layer import Layer
from softmax import Softmax
from datetime import datetime


class Model:
    def __init__(self, word_dim, hidden_dim=100, truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.truncate = truncate
        self.U = np.random.uniform(-np.sqrt(1.0 / word_dim),
                                   np.sqrt(1.0 / word_dim),
                                   (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1.0 / hidden_dim),
                                   np.sqrt(1.0 / hidden_dim),
                                   (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1.0 / hidden_dim),
                                   np.sqrt(1.0 / hidden_dim),
                                   (word_dim, hidden_dim))

    def forward_pass(self, x):
        time_steps = len(x)
        layers = []
        prev = np.zeros(self.hidden_dim)

        for i in range(time_steps):
            layer = Layer()
            _input = np.zeros(self.word_dim)
            _input[x[i]] = 1
            layer.forward(_input, self.U, self.W, self.V, prev)
            prev = layer.act
            layers.append(layer)

        return layers

    def predict(self, x):
        output = Softmax()
        layers = self.forward_pass(x)

        return [np.argmax(output.predict(layer.mulV)) for layer in layers]

    def loss(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_pass(x)
        loss = 0.0

        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulV, y[i])

        return loss / float(len(y))

    def total_loss(self, x, y):
        loss = 0.0
        for i in range(len(y)):
            loss += self.loss(x[i], y[i])

        return loss / float(len(y))

    def backward_pass(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_pass(x)

        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        time_steps = len(x)
        prev_t = np.zeros(self.hidden_dim)
        diff = np.zeros(self.hidden_dim)

        for i in range(0, time_steps):
            dmulV = output.diff_scores(layers[i].mulV, y[i])
            _input = np.zeros(self.word_dim)
            _input[x[i]] = 1
            dU_i, dW_i, dV_i, dprev_act = layers[i].backward(
                _input, self.U, self.W, self.V, prev_t, diff, dmulV)
            prev_t = layers[i].act
            dmulV = np.zeros(self.word_dim)
            for j in range(i - 1, max(-1, i - self.truncate - 1), -1):
                _input = np.zeros(self.word_dim)
                _input[x[j]] = 1
                prev_j = np.zeros(
                    self.hidden_dim) if j == 0 else layers[j - 1].act
                dU_j, dW_j, dV_j, dprev_act = layers[i].backward(
                    _input, self.U, self.W, self.V, prev_j, dprev_act, dmulV)
                dU_i += dU_j
                dW_i += dW_j
            dV += dV_i
            dU += dU_i
            dW += dW_i

        return (dU, dW, dV)

    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.backward_pass(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW

    def train(self,
              x,
              y,
              learning_rate=0.005,
              nb_epoch=100,
              evaluate_loss_after=2):
        examples_seen = 0
        losses = []

        for epoch in range(nb_epoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.total_loss(x, y)
                losses.append((examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" %
                      (time, examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example
            for i in range(len(y)):
                self.sgd_step(x[i], y[i], learning_rate)
                examples_seen += 1
        return losses