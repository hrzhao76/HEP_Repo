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
from lightgbm import LGBMClassifier



log_levels = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
}
def calculte_binerror(values, weights, bins):
    #### Return sqrt of sum of weights
    binned_idx = np.digitize(values, bins=bins) - 1
    n_bins = len(bins) - 1  
    err = np.zeros(n_bins)
    for i, bin_left in enumerate(bins[:-1]):
        weights_at_bin = weights[np.where(binned_idx == i)[0]]
        err[i] = np.sqrt(np.sum(np.power(weights_at_bin, 2)))
    
    return err

def get_gbdt_model(n_estimators, learning_rate, max_depth, max_features=None, min_samples_leaf=1):
    # bdt = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, validation_fraction=1/9,
    #                              max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, random_state=42, verbose=1)
    bdt = LGBMClassifier(n_estimators = n_estimators, objective ='binary', random_state=42)

    return bdt

def plot_decision_func(X, y_decisions, target, output_path):
    y_test_gluon_id = np.where(target==1)[0]
    y_test_quark_id = np.where(target==0)[0]
    weights_gluon = X.iloc[y_test_gluon_id, X.columns.get_loc('event_weight')]
    weights_quark = X.iloc[y_test_quark_id, X.columns.get_loc('event_weight')]
    bins_scores = np.linspace(-5, 5, 101)

    fig, ax = plt.subplots()
    plt.hist(y_decisions[y_test_gluon_id], weights=weights_gluon, 
            bins= bins_scores, alpha=0.5, color = 'blue',
            label='gluon'+f"_num: {len(y_test_gluon_id)}, sum. weights: {np.sum(weights_gluon):.2f}") # add the weights! 
    plt.hist(y_decisions[y_test_quark_id], weights=weights_quark, 
            bins= bins_scores, alpha=0.5, color = 'red',
            label='quark'+f"_num: {len(y_test_quark_id)}, sum. weights: {np.sum(weights_quark):.2f}")
    plt.legend(loc='upper left')
    plt.text(0.05, 0.75, f"num: {len(X)}, \nsum. of weights:{np.sum(X['event_weight']):.2f}", transform=plt.gca().transAxes)
    plt.xlabel("BDT Decision Function")
    plt.ylabel("Weighted yield of jets")
    plt.title(r"New Training with event weight")
    plt.savefig(os.path.join(output_path, "GBDT_Training_decision_function.png"))
    plt.close()

def plot_ROC(y_decisions, y_tmva, y_ntrk, target, X_weight, features, output_path):
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(target, y_decisions, sample_weight = X_weight)
    fpr_tmva, tpr_tmva, thresholds_tmva = roc_curve(target, y_tmva, sample_weight = X_weight)
    fpr_ntrk, tpr_ntrk, thresholds_ntrk =  roc_curve(target, y_ntrk, sample_weight = X_weight)

    roc_auc = auc(fpr, tpr)
    roc_auc_tmva = auc(fpr_tmva, tpr_tmva)
    roc_auc_ntrk = auc(fpr_ntrk, tpr_ntrk)

    fig, ax = plt.subplots()
    plt.plot(1-fpr, tpr, lw=1, label='ROC_NewTraining (area = %0.3f)'%(roc_auc))
    plt.plot(1-fpr_tmva, tpr_tmva, lw=1, label='ROC_oldBDT (area = %0.3f)'%(roc_auc_tmva))
    plt.plot(1-fpr_ntrk, tpr_ntrk, lw=1, label='ROC_Ntrk (area = %0.3f)'%(roc_auc_ntrk))

    plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC with features:{features}'+r" physical $p_{T}$ weight")
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig(os.path.join(output_path, "ROC.png"))
    plt.close()

def plot_ROC_train_test(y_decisions, y_tmva, y_ntrk, target, X_weight, target_dev, y_dev_decisions, X_dev_weight, features, output_path):
    # Compute ROC curve and area under the curve
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(target, y_decisions, sample_weight = X_weight)
    fpr_dev, tpr_dev, thresholds_dev = roc_curve(target, y_decisions, sample_weight = X_weight)
    fpr_tmva, tpr_tmva, thresholds_tmva = roc_curve(target, y_tmva, sample_weight = X_weight)
    fpr_ntrk, tpr_ntrk, thresholds_ntrk =  roc_curve(target, y_ntrk, sample_weight = X_weight)

    roc_auc = auc(fpr, tpr)
    roc_auc_tmva = auc(fpr_tmva, tpr_tmva)
    roc_auc_ntrk = auc(fpr_ntrk, tpr_ntrk)

    plt.plot(1-fpr, tpr, lw=1, label='ROC_NewTraining (area = %0.3f)'%(roc_auc))
    plt.plot(1-fpr_tmva, tpr_tmva, lw=1, label='ROC_oldBDT (area = %0.3f)'%(roc_auc_tmva))
    plt.plot(1-fpr_ntrk, tpr_ntrk, lw=1, label='ROC_Ntrk (area = %0.3f)'%(roc_auc_ntrk))

    plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC with features:{features}'+r" physical $p_{T}$ weight")
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig(os.path.join(output_path, "ROC_train_test.png"))
    plt.close()

def plot_overtraining_validation(X_dev, X_test, y_dev, y_test, y_dev_decisions, y_test_decisions, output_path):
    y_test_decisions_gluon = y_test_decisions[np.where(y_test==1)]
    y_test_decisions_quark = y_test_decisions[np.where(y_test==0)]
    y_dev_decisions_gluon = y_dev_decisions[np.where(y_dev==1)]
    y_dev_decisions_quark = y_dev_decisions[np.where(y_dev==0)]
    decisions = []
    decisions.append(y_dev_decisions_gluon)
    decisions.append(y_dev_decisions_quark)
    decisions.append(y_test_decisions_gluon)
    decisions.append(y_test_decisions_quark)
    custom_bins = np.linspace(-5, 5, 101)
    n_bins = len(custom_bins) - 1
    custom_bin_width = (custom_bins[1] - custom_bins[0])
    custom_bin_center = (custom_bins[:-1] + custom_bins[1:]) / 2

    fig, ax = plt.subplots()
    event_weight_idx = X_dev.columns.get_loc('event_weight')
    ax.hist(decisions[0], weights = X_dev.iloc[np.where(y_dev==1)[0], event_weight_idx],
                color='b', alpha=0.5, bins=custom_bins,
                histtype='stepfilled', density=True,
                label='Gluon (train)')
    ax.hist(decisions[1], weights = X_dev.iloc[np.where(y_dev==0)[0], event_weight_idx],
                color='r', alpha=0.5, bins=custom_bins,
                histtype='stepfilled', density=True,
                label='Quark (train)')

    hist, bins = np.histogram(decisions[2], weights=X_test.iloc[np.where(y_test==1)[0], event_weight_idx],
                                bins=custom_bins,  density=True)
    scale = len(decisions[2]) / sum(hist) 
    # err = np.sqrt(hist * scale) / scale # Used for equal weight
    err = calculte_binerror(values=decisions[2], weights=X_test.iloc[np.where(y_test==1)[0], event_weight_idx].values, bins=custom_bins) / scale

    plt.errorbar(custom_bin_center, hist, yerr=err, fmt='o', c='b', label='Gluon (test)')

    hist, bins = np.histogram(decisions[3], weights=X_test.iloc[np.where(y_test==0)[0], event_weight_idx],
                                bins=custom_bins, density=True)
    scale = len(decisions[3]) / sum(hist)
    # err = np.sqrt(hist * scale) / scale
    err = calculte_binerror(values=decisions[3], weights=X_test.iloc[np.where(y_test==0)[0], event_weight_idx].values, bins=custom_bins) / scale

    plt.errorbar(custom_bin_center, hist, yerr=err, fmt='o', c='r', label='Quark (test)')
    plt.xlabel("GBDT decision function output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.title("Overtraining test")
    plt.savefig(os.path.join(output_path, "overtrain_validation.png"))
    plt.close()

def train(sample_path, clf_model, training_vars, training_weights, output_path):
    sample = pd.read_pickle(sample_path)

    X = sample.iloc[:, :-1]

    target_idx = sample.columns.get_loc('target')
    y = sample.iloc[:, target_idx]

    X_dev,X_test, y_dev,y_test = train_test_split(X, y, test_size=0.1, random_state=456)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.1/0.9, random_state=789)

    logging.debug(X_dev[training_vars].head())
    logging.debug(X_dev[training_weights].head())

    if isinstance(clf_model, LGBMClassifier):
        clf_model.fit(X_dev[training_vars], y_dev, sample_weight=X_dev[training_weights].to_numpy().flatten())
        logging.info(f"Feature importance: {clf_model.feature_importances_}")

        y_dev_decisions=clf_model.predict(X_dev[training_vars], raw_score = True)
        y_test_decisions = clf_model.predict(X_test[training_vars], raw_score = True)
        logging.info("Plotting distribution on test dataset...")
        plot_decision_func(X_test, y_test_decisions, y_test, output_path)

        logging.info("Plotting ROC on test dataset...")
        plot_ROC(y_decisions=y_test_decisions, y_tmva=X_test.iloc[:,X_test.columns.get_loc('jet_trackBDT')], 
                 y_ntrk=X_test.iloc[:,X_test.columns.get_loc('jet_nTracks')], target=y_test, 
                 X_weight=X_test['event_weight'], features=" 4 vars", output_path=output_path)

        logging.info("Plotting overtraining validation plot...")
        plot_overtraining_validation(X_dev=X_dev, X_test=X_test, y_dev=y_dev, y_test=y_test, 
                                    y_dev_decisions=y_dev_decisions, y_test_decisions=y_test_decisions, 
                                    output_path=output_path)


    return clf_model

def main(output_path, sample_path, training_vars, training_weights, **kwargs):
    try:
        n_estimators = kwargs['n_estimators']
        learning_rate = kwargs['learning_rate']
        max_depth = kwargs['max_depth']
    except KeyError:
        logging.ERROR("keyword not found in {kwargs}")

    clf_model = get_gbdt_model(n_estimators, learning_rate, max_depth)
    clf = train(sample_path=sample_path, clf_model=clf_model, training_vars=training_vars, training_weights=training_weights,
                output_path=output_path)
    pickle.dump(clf, open(os.path.join(output_path, "clf.pkl"), 'wb'))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='the output folder path', type=str)
    parser.add_argument('--sample-path', help='training sample path', type=str)
    parser.add_argument('--training-vars', nargs='+', help='training features', default=['jet_pt', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1'])
    parser.add_argument('--training-weights', help='training weight', type=str, default='flatpt_weight')

    parser.add_argument('--n-estimators', help='number of estimators', type=int, default=100)
    parser.add_argument('--learning-rate', help='learning rate', type=float, default=1.0)
    parser.add_argument('--max_depth', help='max depth', type=float, default=2)

    parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=0,
                    help="Verbosity (between 1-4 occurrences with more leading to more "
                         "verbose logging). CRITICAL=0, ERROR=1, WARN=2, INFO=3, "
                         "DEBUG=4")

    args = parser.parse_args()

    output_folder = Path(args.output_path)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=args.output_path + '/BDT_train.log', filemode='w', level=log_levels[args.verbosity], 
                        format='%(asctime)s   %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


    main(output_path=args.output_path, sample_path=args.sample_path, 
    training_vars=args.training_vars, training_weights=args.training_weights,
    n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth)

    logging.info('Done')







