import random
import shutil

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import uproot 
import awkward as ak
from pathlib import Path
import os

def cut(branches):
    cutflow = []
    cutflow.append(len(branches))
    selected_evt_idx = np.ones(len(branches))

    cuts =  branches["jet_fire"] == 1, \
            ak.count(branches["jet_pt"], axis=1) > 1, \
            branches["jet_pt"][:,0] > 500 * 1e3,\
            branches["jet_pt"][:,0] < 2000 * 1e3, \
            np.abs(branches["jet_eta"][:,0]) < 2.1,\
            np.abs(branches["jet_eta"][:,1]) < 2.1,\
            branches["jet_pt"][:,0] / branches["jet_pt"][:,1] < 1.5

    for i, cut_evt_idx in enumerate(cuts):
        selected_evt_idx = np.logical_and(selected_evt_idx, cut_evt_idx)
        cutflow.append(ak.sum(selected_evt_idx))
    
    branches = branches[selected_evt_idx]
    return branches, cutflow



sample_folder_path = Path("/home/dejavu/Projects/Samples/qgcal/")
unmerged_files = sorted(sample_folder_path.glob("data*_13TeV.period*_merged.root"))
with uproot.recreate("merged_data.root") as merged_file:
    file = unmerged_files[0]
    file_name = file.stem

    branch_names = ["jet_fire", "jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]
    print(f"Merging {file_name}.root ")

    file=uproot.open(file)
    branches = file["nominal"].arrays(branch_names, library='ak')


    branches, cutflow = cut(branches)
    branches = branches[branch_names[1:]][:,:2] # only keep 2 jets in each event 
    total_weight = np.ones(len(branches))
    branches['total_weight']=total_weight

    merged_file['nominal'] = {branch : branches[branch] for branch in branches.fields}
    
    for file in unmerged_files[1:]:
        file_name = file.stem
        print(f"Merging {file_name}.root ")

        file=uproot.open(file)
        branches = file["nominal"].arrays(branch_names, library='ak')

        branches, cutflow = cut(branches)
        branches = branches[branch_names[1:]][:,:2] # only keep 2 jets in each event 
        total_weight = np.ones(len(branches))
        branches['total_weight']=total_weight
        merged_file["nominal"].extend({branch : branches[branch] for branch in branches.fields})

