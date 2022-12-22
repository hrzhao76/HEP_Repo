import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import uproot 
import awkward as ak
from pathlib import Path

from typing import Dict, List 
import re
import pickle
from tqdm import tqdm

from sklearn.model_selection import train_test_split


training_vars = ['jet_pt', 'jet_eta', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1']
all_vars = training_vars + ['total_weight', 'flatpt_weight']
n_jets = 2_000_000

sample_alljets_path = '../../samples/BDT_training/sample_2M_w_flatpt.pkl'
with open(sample_alljets_path, 'rb') as f:
    sample_2Mjets = pd.read_pickle(f)

X = sample_2Mjets.iloc[:, :-1]
y = sample_2Mjets.iloc[:, -1]


X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=0.1, random_state=456)
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.1/0.9, random_state=789)

