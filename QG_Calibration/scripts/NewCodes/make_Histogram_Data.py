import argparse
import numpy as np

import uproot 
import awkward as ak
from pathlib import Path

from typing import Dict, List 
import re
import pickle

label_pt_leadingtype = ["LeadingJet", "SubLeadingJet"]
label_eta = ["Forward", "Central"]
label_type = ["Data"] # For Data, no other type info, but must match split_jet_type()
label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]


def Initialize_Map():
    Map = {
        "values": {},
        "weights": {}
    }

    for l_pt in label_pt_bin[:-1]:
        for l_pt_leading_type in label_pt_leadingtype:
            for l_eta in label_eta:
                for l_type in label_type:
                    Map["weights"][f"{str(l_pt)}_{l_pt_leading_type}_{l_eta}_{l_type}"] = np.array([], dtype = np.float32)
                    for l_var in label_var:
                        Map["values"][f"{str(l_pt)}_{l_pt_leading_type}_{l_eta}_{l_type}_{l_var}"] = np.array([], dtype = np.float32)

    assert len(label_pt_bin[:-1]) * len(label_pt_leadingtype) * len(label_eta) * len(label_type) == len([*Map["weights"].keys()])
    assert len(label_pt_bin[:-1]) * len(label_pt_leadingtype) * len(label_eta) * len(label_type) * len(label_var) == len([*Map["values"].keys()])
    return Map 

def Minitree2Hist_Data(input_folder, period, reweight_var = None, reweight_factor = None):
    if not reweight_var == None and reweight_factor == None:
        raise Exception(f"You specify reweight_type to be {reweight_var} but reweight_factor is not provided! ")

    if not reweight_var == None and not reweight_var in ["ntrk", "bdt"]:
        raise Exception(f"You specify reweight_type to be {reweight_var} but not supported! ")
   
    HistMap = Initialize_Map()

    branch_names = ["jet_fire", "jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]
    branch_names_tobesaved = ["jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]

    for i, dir in enumerate(input_folder):
        print(dir)
        
        for file in sorted(dir.glob("*.minitrees.root")):
            print(f"        Doing on file {file.stem}")
            try:
                branches_before_cut = uproot.open(file)["nominal"].arrays(branch_names, library='ak')
            except: 
                print(f"        Doing on file {file.stem}, but the tree is empty")
            
            if len(branches_before_cut) == 0:
                print(f"        Doing on file {file.stem}, but it's empty")
                continue 

            branches_after_cut = Apply_Cuts(branches_before_cut = branches_before_cut, period = period)
            if len(branches_after_cut) == 0:
                continue 

            branches_to_be_saved = branches_after_cut[branch_names_tobesaved][:,:2]
            leading_jets = branches_to_be_saved[branch_names_tobesaved][:,0].to_numpy()
            subleading_jets = branches_to_be_saved[branch_names_tobesaved][:,1].to_numpy()

            leading_jets["jet_pt"] = leading_jets["jet_pt"]/1000
            subleading_jets["jet_pt"] = subleading_jets["jet_pt"]/1000

            leading_jets = np.dstack([leading_jets[field] for field in leading_jets.dtype.names])
            subleading_jets = np.dstack([subleading_jets[field] for field in subleading_jets.dtype.names])

            total_weight = branches_after_cut["total_weight"].to_numpy()
            leading_jets = np.concatenate((leading_jets, np.broadcast_to(total_weight[None, : , None], (leading_jets.shape[:2] + (1,)))), axis = 2)
            subleading_jets = np.concatenate((subleading_jets, np.broadcast_to(total_weight[None, : , None], (subleading_jets.shape[:2] + (1,)))), axis = 2)

            dijets = np.concatenate((leading_jets,subleading_jets), axis = 0)
            dijets = np.swapaxes(dijets, 0, 1) # the format is (n_events, 2, 9)
            
            HistMap = Split_jets_Data(HistMap = HistMap, dijets_array = dijets, reweight_var = reweight_var, reweight_factor = reweight_factor)

    return HistMap

    
        
def Apply_Cuts(branches_before_cut: ak.highlevel.Array, period: str):

    total_weight_before_cut = ak.ones_like(branches_before_cut["jet_fire"])

    if period == "18":
        total_weight_before_cut = total_weight_before_cut * 58.45/39.91
    branches = ak.with_field(base = branches_before_cut, what = total_weight_before_cut, where = "total_weight")

    # event trigger selection 
    event_trigger_idx = branches["jet_fire"] == 1
    branches = branches[event_trigger_idx]

    # pT cut 
    pt_min_cut_idx = branches["jet_pt"][:,0] > 500000
    branches = branches[pt_min_cut_idx]

    pt_max_cut_idx = branches["jet_pt"][:,0] < 2000000 
    branches = branches[pt_max_cut_idx]

    # ratio < 1.5
    branches = branches[branches["jet_pt"][:,0]/branches["jet_pt"][:,1] < 1.5]

    # eta cut 
    branches = branches[np.abs(branches["jet_eta"][:,0]) < 2.1]
    branches = branches[np.abs(branches["jet_eta"][:,1]) < 2.1]

    # weight cut, for weight > 100 set it to 1 
    # Change to discard the large weight! 
    # Should be no effects for Data 
    branches = branches[np.abs(branches["total_weight"]) < 100] 

    return branches
    
def Split_jets_Data(HistMap, dijets_array, reweight_var = None, reweight_factor = None):
    n_jets = 0
    splited_pt_eta_jets = split_pt_eta_jet(dijets_array)

    for ki, splited_pt_eta_jet in splited_pt_eta_jets.items():
        # ki is the key, like LeadingJet_Forward
        n_jets += len(splited_pt_eta_jet)
        
        splited_jet_pt_bins = split_jet_pt(splited_pt_eta_jet)

        for kj, splited_jet_pt_bin in splited_jet_pt_bins.items():
            # kj is the pt, like 500 
            if splited_jet_pt_bin.shape[0] == 0:
                continue 
            
            assert label_eta[1] == "Central" # label_eta[1] should be the Central 
            if reweight_var in ["ntrk", "bdt"] and ki.__contains__(label_eta[1]):  # Doing reweighting for Central jets 
                # breakpoint()
                # Do reweighting here 
                if reweight_var == "ntrk":
                    reweight_var_idx = 2
                elif reweight_var == "bdt":
                    reweight_var_idx = 5

                reweight_var_bins = GetHistBin(reweight_var)

                reweighting_at_pt = reweight_factor[str(kj)]
                inds = np.digitize(splited_jet_pt_bin[:, reweight_var_idx], reweight_var_bins)
                inds[inds > (len(reweight_var_bins) - 1)] = len(reweight_var_bins) - 1  # Ensure that > the boundary are proper dealt
                # e.g. ntrk = 75 > 60 that is outside of reweighting_at_pt
                # reweight_var_bins[inds] will out of range for ntrk; just modify the inds to be the last bin 

                splited_jet_pt_bin[:, -1] = splited_jet_pt_bin[:, -1] * reweighting_at_pt[inds-1]
 
                
            splited_pt_eta_jets_types = split_jet_type(splited_jet_pt_bin)

            for kk, jet_type in splited_pt_eta_jets_types.items():
                if jet_type.shape[0] == 0:
                    continue 

                histogram_name = f"{str(kj)}_{ki}_{kk}"
                HistMap["weights"][histogram_name] = np.append(HistMap["weights"][histogram_name], jet_type[:, -1])
                
                for kl, l_var in enumerate(label_var): 
                    HistMap["values"][histogram_name+f"_{l_var}"] = np.append(HistMap["values"][histogram_name+f"_{l_var}"], jet_type[:, kl])
                

    assert n_jets == len(dijets_array)*2 # Check if jets are splited correctly 
    return HistMap

def split_pt_eta_jet(jets):
    # divide jets into 4 regions 
    forward_idx = np.argmax(np.abs(jets[:,:,1]), axis=1) # compare abs eta of jets inside events
    central_idx = -1*forward_idx+1
    
    leading_forward_idx = forward_idx == 0 # leading forward 
    subleading_forward_idx = forward_idx == 1 # subleading forward 

    leading_central_idx = central_idx == 0 # leading central 
    subleading_central_idx = central_idx == 1 # subleading central 

    return { f"{label_pt_leadingtype[0]}_{label_eta[0]}" : jets[leading_forward_idx, 0, :], 
             f"{label_pt_leadingtype[0]}_{label_eta[1]}" : jets[leading_central_idx, 0, :],  
             f"{label_pt_leadingtype[1]}_{label_eta[0]}" : jets[subleading_forward_idx, 1, :], 
             f"{label_pt_leadingtype[1]}_{label_eta[1]}" : jets[subleading_central_idx, 1, :]}

def split_jet_type(jets):
    data_idx = np.where(jets[:,6]==-9999)[0]

    all_list = [jets[data_idx]]
    return_map = {}
    for i in range(len(label_type)):
        return_map[label_type[i]] = all_list[i]

    return return_map 

def split_jet_pt(jets):
    splited_jet_pt = {}
    for i, pt_start in enumerate(label_pt_bin[:-1]):
        pt_selected_idx = np.where((jets[:,0] >= pt_start) & (jets[:,0] < label_pt_bin[i+1]))[0]
        splited_jet_pt[pt_start] = jets[pt_selected_idx]

    return splited_jet_pt


###### define functions
def GetHistBin(histogram_name: str):
    if 'pt' in histogram_name:
        return np.linspace(0, 2000, 61)
    elif 'eta' in histogram_name:
        return np.linspace(-2.5, 2.5, 51)
    elif 'ntrk' in histogram_name:
        return np.linspace(0, 60, 61)
    elif 'bdt' in histogram_name:
        return np.linspace(-0.8, 0.7, 61)
    elif 'width' in histogram_name:
        return np.linspace(0, 0.4, 61)
    elif 'c1' in histogram_name:
        return np.linspace(0, 0.4, 61)
    elif 'newBDT' in histogram_name:
        return np.linspace(-0.8, 0.7, 61)

def WriteHistRootFile(HistMap, output_file_name, TDirectory_name = "NoReighting"):
    print(f"Writing Histogram to the file: {output_file_name}")
    with uproot.update(output_file_name) as foutput:
        for weights_hist_name in HistMap["weights"].keys():
    
            for l_var in label_var:
                values_hist_name = weights_hist_name + f"_{l_var}"
                bin_edges = GetHistBin(histogram_name = values_hist_name)
                histogram_contents = np.histogram(a = HistMap["values"][values_hist_name], weights = HistMap["weights"][weights_hist_name], 
                                        bins = bin_edges, range = (bin_edges[0], bin_edges[-1]))
                foutput[f"{TDirectory_name}/{values_hist_name}"] = histogram_contents

                nbins = len(bin_edges) - 1 
                sum_w2_at_var = np.zeros((nbins,), dtype = np.float32)
                inds = np.digitize(x = HistMap["values"][values_hist_name], bins = bin_edges)
                inds = inds - 1
                for i in range(nbins):
                    weights_at_bin = HistMap["weights"][weights_hist_name][np.where(inds == i)[0]]
                    sum_w2_at_var[i] = np.sum(np.power(weights_at_bin, 2))

                histogram_err = np.histogram(a = bin_edges[:-1], weights = sum_w2_at_var, 
                                bins = bin_edges, range = (bin_edges[0], bin_edges[-1]))
                foutput[f"{TDirectory_name}/{values_hist_name}_err"] = histogram_err

def WritePickleFile(HistMap, pkl_file_name):
    print(f"Writing jets info to the pickle file: {pkl_file_name}")
    with open(pkl_file_name, "wb") as out_pkl:
        pickle.dump(HistMap, out_pkl)

def Make_Histogram_Data(sample_folder_path, period, output_path, reweighting_file_path):
    if not period in ['1516', '17', '18']:
        raise Exception(f"Period {period} not in supported periods. Currently supported: ['1516', '17', '18']")

    if period == '1516':
        period_data = sorted(sample_folder_path.rglob(f"*data{period[:2]}_13TeV.period*.physics_Main_minitrees.root")) + sorted(sample_folder_path.rglob(f"*data{period[2:]}_13TeV.period*.physics_Main_minitrees.root"))
        # Debug use
        # period_data = sorted(sample_folder_path.rglob(f"*data{period[:2]}_13TeV.periodD.physics_Main_minitrees.root")) + sorted(sample_folder_path.rglob(f"*data{period[2:]}_13TeV.periodA.physics_Main_minitrees.root"))

    else:
        period_data = sorted(sample_folder_path.rglob(f"*data{period}_13TeV.period*.physics_Main_minitrees.root"))

    output_file_name = output_path.as_posix() + f"/dijet_data_{period}"

    output_root_file =  output_file_name + ".root"
    output_pickle_file =  output_file_name + ".pkl"
    HistMap = Minitree2Hist_Data(input_folder = period_data, period = period)  

    uproot.recreate(output_root_file)
    WriteHistRootFile(HistMap, output_root_file, TDirectory_name = "NoReweighting")
    WritePickleFile(HistMap, output_pickle_file)  

    period_mc = {
        '1516': 'A',
        '17': 'D',
        '18': 'E'
    }

    ntrk_quark_reweighting_file = reweighting_file_path / f"dijet_pythia_mc16{period_mc[period]}_ntrk_reweighting_quark_factor.pkl"
    ntrk_gluon_reweighting_file = reweighting_file_path / f"dijet_pythia_mc16{period_mc[period]}_ntrk_reweighting_gluon_factor.pkl"

    bdt_quark_reweighting_file = reweighting_file_path / f"dijet_pythia_mc16{period_mc[period]}_bdt_reweighting_quark_factor.pkl"
    bdt_gluon_reweighting_file = reweighting_file_path / f"dijet_pythia_mc16{period_mc[period]}_bdt_reweighting_gluon_factor.pkl"

    reweight_vars = ['ntrk', 'bdt']
    reweight_factors = ['Quark', 'Gluon']
    reweight_files = {
        'ntrk_Quark':ntrk_quark_reweighting_file,
        'ntrk_Gluon':ntrk_gluon_reweighting_file,
        'bdt_Quark':bdt_quark_reweighting_file,
        'bdt_Gluon':bdt_gluon_reweighting_file
    }
    # Doing reweighting version 
    for reweight_var in reweight_vars:
        for reweight_factor in reweight_factors[0:1]:
            reweight_file = reweight_files[f'{reweight_var}_{reweight_factor}']

            if Path(reweight_file).exists():
                print(f"Reweighting file {reweight_file} found, read reweighting factor from the reweigthing file. ")
                with open(reweight_file, "rb") as file:
                    reweighting_factor = pickle.load(file)
            else:
                print(f"Reweighting file {reweight_file} not found, read reweighting factor from the histogram root file. ")

            HistMap_ntrk_reweighting_quark_factor = Minitree2Hist_Data(input_folder = period_data, period = period, reweight_var = reweight_var, reweight_factor = reweighting_factor)
            WriteHistRootFile(HistMap_ntrk_reweighting_quark_factor, output_root_file, TDirectory_name = f"{reweight_var}_Reweighting_{reweight_factor}_Factor")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script converts data minitrees to histogram in root and pkl.')
    parser.add_argument('--path', help='The path to the minitree files')
    parser.add_argument('--period', help='The Data period', choices=['1516', '17', '18'])
    parser.add_argument('--output-path', help='Output path')
    parser.add_argument('--reweight-file-path', help='Reweighting file path. Expect 4 reweighting files.')
    args = parser.parse_args()

    minitrees_folder_path = Path(args.path)
    period = args.period
    output_path = Path(args.output_path)
    hist_folder_path = minitrees_folder_path.parent / (minitrees_folder_path.stem + "_hist")

    reweighting_file_path = Path(args.reweight_file_path)
    if not reweighting_file_path.exists():
        raise Exception(f"the input reweight file path {reweighting_file_path.as_posix()} not found. ")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    Make_Histogram_Data(sample_folder_path=minitrees_folder_path, period=period, output_path = output_path, reweighting_file_path = reweighting_file_path)



