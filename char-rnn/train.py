import time
import os
import six
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--data_dir',
    type=str,
    default='/data/tiny-shakespeare.txt',
    help='Data directory containing a txt file as input with training examples'
)

parser.add_argument(
    '--save_dir',
    type=str,
    default='save',
    help='directory to store checkpointed models')

parser.add_argument(
    '--model',
    type=str,
    default='lstm',
    help='Model types supported: LSTM and RNN')

parser.add_argument(
    '--rnn_size', type=int, default=64, help='Size of the RNNs hidden state')

parser.add_argument(
    '--num_layers',
    type=int,
    default=2,
    help='Specifies the n umber of layers in the RNN')

parser.add_argument(
    '--sequence_len',
    type=int,
    default=25,
    help=
    'RNN sequence length. Number of timesteps required by the RNN to unroll.')

parser.add_argument(
    '--batch_size',
    type=int,
    default='25',
    help=
    'Minibatch size. Number of sequences propagated through the network in parallel.'
)

parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help=
    'Number of epochs. Number of full passes through the training examples.')

parser.add_argument(
    '--grad_clip',
    type=float,
    default=5.,
    help='Start clipping gradients at this value')

parser.add_argument(
    '--learning_rate', type=float, default=0.05, help='Learning rate')

parser.add_argument(
    '--decay_rate', type=float, default=0.9, help='Decay rate for RMSprop')

parser.add_argument(
    '--output_keep_prob',
    type=float,
    default=1.,
    help='Probability of keeping weights in the hidden layer')

parser.add_argument(
    '--input_keep_prob',
    type=float,
    default=1.,
    help='Probability of keeping weights in the input layer')

args = parser.parse_args()