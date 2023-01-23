# %%
### Use PyROOT to conver flatten events in pkl to root 
### Prapare for the acts fitting

# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ROOT
import uproot 
import awkward as ak
from pathlib import Path
import os, sys
from tqdm import tqdm
from array import array 

# %%
sample_folder_path = Path(sys.argv[1])
files = sorted(sample_folder_path.glob("*.pkl"))

# %%
file = files[0]
data = pd.read_pickle(file)

# %%
outputFile = ROOT.TFile(f"{sample_folder_path.as_posix()}/SPVCNN_outputs.root", "recreate")
outputTree = ROOT.TTree("SPVCNN_Cluster", "SPVCNN_Cluster")

m_eventID = array('i', [0])
reco_vtx_x0_seed = ROOT.std.vector[float]()
reco_vtx_y0_seed = ROOT.std.vector[float]()
reco_vtx_z0_seed = ROOT.std.vector[float]()

truth_vtx_fitted_trk_d0 = ROOT.std.vector[float]()
truth_vtx_fitted_trk_z0 = ROOT.std.vector[float]()
truth_vtx_fitted_trk_phi = ROOT.std.vector[float]()
truth_vtx_fitted_trk_theta = ROOT.std.vector[float]()
truth_vtx_fitted_trk_qp = ROOT.std.vector[float]()
truth_vtx_fitted_trk_time = ROOT.std.vector[float]()

truth_vtx_fitted_trk_err_d0 = ROOT.std.vector[float]()
truth_vtx_fitted_trk_err_z0 = ROOT.std.vector[float]()
truth_vtx_fitted_trk_err_phi = ROOT.std.vector[float]()
truth_vtx_fitted_trk_err_theta = ROOT.std.vector[float]()
truth_vtx_fitted_trk_err_qp = ROOT.std.vector[float]()
truth_vtx_fitted_trk_err_time = ROOT.std.vector[float]()

truth_vtx_fitted_trk_SPVCNN_vtxID = ROOT.std.vector[int]()

fitted_trk_list = [truth_vtx_fitted_trk_d0, truth_vtx_fitted_trk_z0, 
                   truth_vtx_fitted_trk_phi, truth_vtx_fitted_trk_theta, 
                   truth_vtx_fitted_trk_qp, truth_vtx_fitted_trk_time,
                   truth_vtx_fitted_trk_err_d0, truth_vtx_fitted_trk_err_z0,
                   truth_vtx_fitted_trk_err_phi, truth_vtx_fitted_trk_err_theta,
                   truth_vtx_fitted_trk_err_qp, truth_vtx_fitted_trk_err_time,
                   truth_vtx_fitted_trk_SPVCNN_vtxID]

fitted_trk_name_list = ['d0', 'z0', 'phi', 'theta', 'qp', 'time', 
                        'err_d0', 'err_z0', 'err_phi', 'err_theta', 'err_qp', 'err_time', 
                        'reco_SPVCNN_vtxID']

reco_vtx_list = [reco_vtx_x0_seed, reco_vtx_y0_seed, reco_vtx_z0_seed]
reco_vtx_name_list = ['x0', 'y0', 'z0'] 

outputTree.Branch("event_id", m_eventID, "eventID/I")

for i, vtx_property in enumerate(reco_vtx_list):
    outputTree.Branch(f"reco_vtx_{reco_vtx_name_list[i]}_seed", vtx_property)



for j, trk_property in enumerate(fitted_trk_list[:-1]):
    outputTree.Branch(f"truth_vtx_fitted_trk_{fitted_trk_name_list[j]}", trk_property)
    
outputTree.Branch("reco_SPVCNN_vtxID", truth_vtx_fitted_trk_SPVCNN_vtxID)


# %%
for file in tqdm(files):
    data = pd.read_pickle(file)
    m_eventID[0] = int(file.stem[-5:])

    # make pred id consecutive 
    new_label = np.zeros_like(data['pred_instance_labels'])
    for i, old_id in enumerate(np.unique(data['pred_instance_labels'])):
        new_label[ data['pred_instance_labels'] == old_id] = i

    n_clusters = np.max(new_label) + 1 
    
    for i in range(n_clusters):
        trk_idxs = np.where(new_label == i)
        reco_vtx_x0_seed.push_back(0)
        reco_vtx_y0_seed.push_back(0)
        reco_vtx_z0_seed.push_back(np.mean(data.iloc[trk_idxs]['z0']))
        
        for trk_idx in trk_idxs[0]:
            for j in range(len(fitted_trk_list)-1):
                fitted_trk_list[j].push_back(data.iloc[trk_idx][fitted_trk_name_list[j]])
            fitted_trk_list[12].push_back(i)

    outputTree.Fill()

    for j, vtx_property in enumerate(reco_vtx_list):
        vtx_property.clear()
    for j, trk_property in enumerate(fitted_trk_list):
        trk_property.clear()

outputTree.Write()
outputFile.Close()
