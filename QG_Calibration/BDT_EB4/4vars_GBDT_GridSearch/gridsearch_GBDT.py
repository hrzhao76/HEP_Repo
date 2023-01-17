# %%
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection 



# %%
training_vars = ['jet_pt', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1']
training_weight = ['flatpt_weight']
gbdt_filename = './gbdt.model'

sample_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/samples/sample_allpt_all_jets.pkl'

label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]

n_estimators = 100
learning_rate = 1.0 
max_depth = 2

# %%
sample = pd.read_pickle(sample_path)

# %%
X = sample.iloc[:, :-1]
y = sample.iloc[:, -1]

X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=0.1, random_state=456)
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.1/0.9, random_state=789)

# %%
event_weight_idx = X.columns.get_loc('event_weight')
equal_weight_idx = X.columns.get_loc('equal_weight')
flatpt_weight_idx = X.columns.get_loc('flatpt_weight')

# %%
bdt = GradientBoostingClassifier(n_estimators=1, learning_rate=learning_rate, 
                                 max_depth=max_depth, random_state=42, verbose=1)

bdt.fit(X_dev[training_vars], y_dev, sample_weight=X_dev[training_weight].to_numpy().flatten())

# %%
param_grid = {"max_depth": [2, 5, 7],
              "n_estimators": [50, 100, 200],
              'learning_rate': [0.1, 0.5, 1.]}


clf = model_selection.GridSearchCV(bdt,
                               param_grid,
                               cv=5,
                               scoring='roc_auc',
                               n_jobs=8,
                               verbose=3)
_ = clf.fit(X_dev[training_vars], y_dev, sample_weight=X_dev[training_weight].to_numpy().flatten())

# %%
print("Best parameter set found on development set:\n")
print(clf.best_estimator_)

# %%
import pickle
with open('cv_results_flat_pt.pkl', 'wb') as f:
    pickle.dump(clf.cv_results_, f)

# %%



