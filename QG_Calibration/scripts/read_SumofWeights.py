import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import uproot 
import awkward as ak
from pathlib import Path
import os

sample_folder_path = Path("/global/cscratch1/sd/hrzhao/Samples/qgcal/unmerged/lxplus")
SumofWeights = []

for file in sorted(sample_folder_path.glob("JZ*_hist.root")):
    hist = uproot.open(file)['histoEventCount']
    SumofWeights.append(hist.values()[0])
    
np.save("SumofWeights.npy", np.array(SumofWeights))