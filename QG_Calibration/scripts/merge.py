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

from read_SumofWeights import SumofWeights

plt.ioff()

sample_folder_path = Path("/global/cscratch1/sd/hrzhao/Samples/qgcal")

xsec = np.array([7.8050E+07, 7.8050E+07, 2.4330E+06, 2.6450E+04, 2.5461E+02, 4.5532E+00, 2.5754E-01, 1.6215E-02, 6.2506E-04, 1.9639E-05])*1E3 # pb
eff = np.array([9.753257E-01, 2.442497E-02, 9.863129E-03, 1.165838E-02, 1.336560E-02, 1.452648E-02, 9.471878E-03, 1.1097E-02, 1.015436E-02, 1.2056E-02])

SumofWeights = np.load("SumofWeights.npy")
SumofWeights_manual = []

lumi_a = 36000 # pb^-1

with uproot.recreate("merged.root") as merged_file:
    file = sorted(sample_folder_path.glob("*.root"))[0]
    file_name = file.stem
    branch_names = ["pu_weight", "mconly_weight", "jet_fire", "jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]

    print(f"Merging {file_name}.root ")

    file=uproot.open(file)
    branches = file["nominal"].arrays(branch_names, library='ak')
    idx = int(file_name[-1])

    total_weight = lumi_a*branches["pu_weight"] *xsec[idx]*eff[idx] / SumofWeights[0]
    SumofWeights_manual.append(ak.sum(branches["mconly_weight"]))

    branches['total_weight']=total_weight
    print(f"{file_name} events: {len(branches)}")
    total_nevents = len(branches)

    # merged_file.mktree("nominal", {branch : branches[branch] for branch in branches.fields})
    # merged_file.mktree("nominal", branches)

    merged_file['nominal'] = {branch : branches[branch] for branch in branches.fields}


    for file in sorted(sample_folder_path.glob("*.root"))[1:]:
        file_name = file.stem
        print(f"Merging {file_name}.root ")

        file=uproot.open(file)
        branches = file["nominal"].arrays(branch_names, library='ak')
        idx = int(file_name[-1])

        total_weight = lumi_a*branches["pu_weight"] *xsec[idx]*eff[idx] / SumofWeights[idx-1]
        SumofWeights_manual.append(ak.sum(branches["mconly_weight"]))
        branches['total_weight']=total_weight
        print(f"{file_name} events: {len(branches)}")
        merged_file["nominal"].extend({branch : branches[branch] for branch in branches.fields})
        # merged_file["nominal"].extend(branches)
        total_nevents += len(branches)

    print(f"total events(counted individually): {total_nevents}")

np.save("SumofWeights_manual.npy", np.array(SumofWeights_manual))

with uproot.open("merged.root") as merged_file:
    nevents= len(merged_file["nominal"]["mconly_weight"].array())
    print(f"total events(counted from merged file): {nevents}")


