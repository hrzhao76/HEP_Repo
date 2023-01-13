import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.set_loglevel("info")
import uproot 
import awkward as ak
from pathlib import Path
import os 

from typing import Dict, List 
import re
import pickle
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from BDT_train import plot_decision_func, plot_ROC

def main(input_path, sample_path, training_vars):
    bdt_model_name = 'clf.pkl'
    sample_path = Path(sample_path)

    with open(os.path.join(input_path, bdt_model_name), 'rb') as f:
        bdt = pickle.load(f)
    
    test_sample = pd.read_pickle(sample_path)

    y_test = test_sample['target']
    X_test = test_sample.drop(['target'], axis = 1)
    
    y_test_decisions = bdt.decision_function(X_test[training_vars])

    output_folder = Path(input_path) / sample_path.stem
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    
    output_path = output_folder.as_posix()

    plot_decision_func(X_test, y_test_decisions, y_test, output_path)

    plot_ROC(y_decisions=y_test_decisions, y_tmva=X_test.iloc[:,X_test.columns.get_loc('jet_trackBDT')], 
                 y_ntrk=X_test.iloc[:,X_test.columns.get_loc('jet_nTracks')], target=y_test, 
                 X_weight=X_test['event_weight'], features=" 4 vars", output_path=output_path)

    return y_test_decisions
                 


    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='the folder path to a bdt model', type=str)
    parser.add_argument('--sample-path', help='test sample path', type=str)
    parser.add_argument('--training-vars', nargs='+', help='training features', default=['jet_pt', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1'])
    

    args = parser.parse_args()
    input = Path(args.input_path)

    if not input.exists():
        raise Exception()
    
    main(args.input_path, sample_path=args.sample_path, training_vars=args.training_vars)
    

    
