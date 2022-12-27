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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split



# %%
training_vars = ['jet_nTracks', 'jet_trackWidth', 'jet_trackC1']
training_weight = ['equal_weight']
sample_path = '../../samples/sample_1500_2p8M_jets.pkl'
sample = pd.read_pickle(sample_path)

X = sample.iloc[:, :-1]
y = sample.iloc[:, -1]

X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=0.1, random_state=456)
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.1/0.9, random_state=789)

event_weight_idx = X.columns.get_loc('event_weight')
equal_weight_idx = X.columns.get_loc('equal_weight')
flatpt_weight_idx = X.columns.get_loc('flatpt_weight')

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

# %%
dt = DecisionTreeClassifier(max_depth=3,
                            min_samples_leaf=0.001,
                            max_features="log2")
bdt = AdaBoostClassifier(dt,
                        algorithm="SAMME",
                        n_estimators=10,
                        learning_rate=0.001,
                        random_state = 42)

# scores = cross_val_score(bdt,
#                         X_dev[training_vars], y_dev,
#                         scoring="roc_auc",
#                         n_jobs=6,
#                         cv=3)
# print(f"test a simple case!")
# print("Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std()))

# %%
print(f"Start GridSearchCV...")

from sklearn import model_selection 
param_grid = {"base_estimator__max_depth": [3, 5, 7, 9],
              "base_estimator__min_samples_leaf": [0.01, 0.001],
              "base_estimator__min_samples_split": [2, 0.01, 0.001],
              "base_estimator__max_features": [None, "log2"],
              "n_estimators": [300, 500, 750, 1000],
              'learning_rate': [0.01, 0.1, 0.5, 1.]}

clf = model_selection.GridSearchCV(bdt,
                               param_grid,
                               cv=3,
                               scoring='roc_auc',
                               n_jobs=16,
                               verbose=3)

_ = clf.fit(X_dev[training_vars] , y_dev)


# %%

print("Best parameter set found on development set:\n")
print(clf.best_estimator_)



# %%
import pickle
with open('cv_results_flat_pt.pkl', 'wb') as f:
    pickle.dump(clf.cv_results_, f)



