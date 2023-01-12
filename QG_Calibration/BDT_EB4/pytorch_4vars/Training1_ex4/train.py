import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from example import * 

_scaler = MinMaxScaler()
#Optimizer
learning_rate = 0.001
epochs = 100
#Layer size
n_hidden1 = 10  # Number of hidden nodes
n_hidden2 = 7
n_output =  1   # Number of output nodes = for binary classifier

use_batch = False
batch_frac = 0.1

train(_scaler, learning_rate, epochs, n_hidden1, n_hidden2, n_output, use_sk_mlp=True, n_hidden3=5)
