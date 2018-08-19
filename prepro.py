import csv
import itertools
import nltk
import numpy as np
import os
import re
import warnings

warnings.filterwarnings("ignore")

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def get_data(filepath, vocab_size=8000):
    _unk_ = "UNKNOWN_TOKEN"
    _start_ = "SENTENCE_START"
    _end_ = "SENTENCE_END"

    print("Reading CSV file...")
    with open(filepath, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = ["{} {} {}".format(_start_, x, _end_) for x in sentences]
    print("Total sentences parsed : {}".format(len(sentences)))

    tokenized_sentences = [tokenize(x) for x in sentences]
    tokenized_sentences = list(filter(lambda x: len(x) > 3, tokenized_sentences))

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Unique word tokens found : {}.".format(len(word_freq.items())))

    vocab = word_freq.most_common(vocab_size - 1)
    idx_to_word = [x[0] for x in vocab]
    idx_to_word.append(_unk_)
    word_to_idx = dict([(w, i) for i, w in enumerate(idx_to_word)])

    print("Using vocabulary size {}".format(vocab_size))

    #for index, group in enumerate(vocab):
    #    print("{}: {}".format(index, group))

    #print("The least frequent word in our vocabulary is '{}' and appeared {} times.".format((vocab[-1][0], vocab[-1][1])))

    for i, x in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_idx else _unk_ for w in x]
        
    print("Example sentence: '{}'".format(sentences[1]))
    print("Example sentence after Pre-processing: '{}'".format(tokenize(sentences[1])))

    #for k, v in enumerate(word_to_idx):
    #    print("{}: {}".format(k, v))

    X_train = np.asarray([[word_to_idx[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_idx[w] for w in sent[1:]] for sent in tokenized_sentences])

    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))

    return X_train, y_train


def main():
    filepath = 'data/reddit-comments-2015-08.csv'
    X_train, y_train = get_data(filepath)

if __name__ == '__main__':
    main()