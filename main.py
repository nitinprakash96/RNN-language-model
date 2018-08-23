import numpy as np

from prepro import get_data
from model import Model

word_dim = 8000
hidden_dim = 100
X_train, y_train = get_data('data/reddit-comments-2015-08.csv', word_dim)

np.random.seed(10)
rnn = Model(word_dim, hidden_dim)

losses = rnn.train(
    X_train[:100],
    y_train[:100],
    learning_rate=0.005,
    nb_epoch=100,
    evaluate_loss_after=2)
