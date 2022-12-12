# just trainig for all the jets in the pt range (500, 2000) GeV
# Grid Search for best parameters 
import random
from re import X

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc


import uproot 
import awkward as ak

file = "/global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/pkls/small_sample_periodA.pkl"

small_sample = pd.read_pickle(file)
small_sample_array = small_sample.to_numpy()

assert small_sample_array.shape == (200000, 10)

X = small_sample_array[:, :-1]
y = small_sample_array[:, -1]

print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")

from sklearn.model_selection import train_test_split

X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=0.1, random_state=456)
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.1/0.9, random_state=789)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

dt = DecisionTreeClassifier(max_depth=3,
                            min_samples_leaf=0.001,
                            max_features="log2")
bdt = AdaBoostClassifier(dt,
                        algorithm="SAMME",
                        n_estimators=800,
                        learning_rate=0.001)

scores = cross_val_score(bdt,
                        X_dev[:,0:5], y_dev,
                        scoring="roc_auc",
                        n_jobs=-1,
                        cv=5)

print("Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std()))

from sklearn import model_selection 
param_grid = {"base_estimator__max_depth": [5, 7, 9],
              "n_estimators": [500, 750, 1000],
              'learning_rate': [0.1, 0.5, 1.]}


clf = model_selection.GridSearchCV(bdt,
                               param_grid,
                               cv=5,
                               scoring='roc_auc',
                               n_jobs=-1,
                               verbose=3)
_ = clf.fit(X_dev[:,:5],y_dev, sample_weight = X_dev[:,7] )

print("Best parameter set found on development set:\n")
print(clf.best_estimator_)
print("Grid scores on a subset of the development set:\n")

import pickle
with open('cv_results_flat_pt.pkl', 'wb') as f:
    pickle.dump(clf.cv_results_, f)
