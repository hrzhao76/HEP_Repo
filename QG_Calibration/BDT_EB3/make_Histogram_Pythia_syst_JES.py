import argparse
import numpy as np
import sys, traceback
import uproot 
import awkward as ak
from pathlib import Path

from typing import Dict, List 
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score


def Read_SumofWeights_Period(sample_folder_path, period):
    """Sum of the first value of unmerged hist.root files and return a numpy array. The length depends on the JZ slices. 

    Args:
        sample_folder_path (Path()): The Path to the rucio downloaded folders.
        period (String): Choose from ["A", "D", "E"], corresponding to periods. 

    Returns:
        Numpy Array: This array is the some of weights from different JZ slices. 
    """
    if not period in ["A", "D", "E"]:
        raise Exception(f'Period {period} not in supported periods. Currently supported: ["A", "D", "E"]')

    period_JZslice = sorted(sample_folder_path.rglob(f"*mc16_13TeV.36470*.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ*WithSW_hist"))
    period_JZ_sum = np.zeros(len(period_JZslice), dtype= float)
    for i, dir in enumerate(period_JZslice):
        print(dir)
        sum_JZ_slice = 0 
        for file in sorted(dir.glob("*.hist-output.root")):
            sum_JZ_slice += uproot.open(file)['histoEventCount'].values()[0]
        
        period_JZ_sum[i] = sum_JZ_slice

    return period_JZ_sum

label_pt_leadingtype = ["LeadingJet", "SubLeadingJet"]
label_eta = ["Forward", "Central"]
label_type = ["Gluon", "Quark", "C_Quark", "B_Quark", "Data", "Other"] # Must match split_jet_type()
label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt", "newbdt"]
label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]

## Debug Use
# label_pt_leadingtype = ["LeadingJet", "SubLeadingJet"]
# label_eta = ["Forward", "Central"]
# label_type = ["Gluon", "Quark", "C_Quark", "B_Quark", "Data", "Other"]
# label_var = ["ntrk"]
# label_pt_bin = [500, 600]
def safe_array_divide(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(numerator, denominator)
        ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
    return ratio

def Get_ReweightingFactor(input_file: str, reweight_var: str ):

    mcfile = uproot.open(input_file)["NoReweighting"]

    quark_factor = {}
    gluon_factor = {}
    for l_pt_bin in label_pt_bin[:-1]:
        HistMap_at_pt_bin = {}
        for i, l_pt_leadingtype  in enumerate(label_pt_leadingtype):
            for j, l_eta in enumerate(label_eta):
                for k, l_type in enumerate(label_type):
                    key = str(l_pt_bin) + "_" + l_pt_leadingtype + "_" + l_eta + "_" + l_type + "_" + reweight_var
                    HistMap_at_pt_bin[key] = mcfile[key].to_numpy()

        if len(HistMap_at_pt_bin) == 0:
            continue 

        Forward_Quark = np.zeros((60))
        Forward_Gluon = np.zeros((60))
        Central_Quark = np.zeros((60))
        Central_Gluon = np.zeros((60))

        for k, v in HistMap_at_pt_bin.items():
            if k.__contains__('Quark') and k.__contains__('Forward'):
                Forward_Quark += v[0]
            elif k.__contains__('Gluon') and k.__contains__('Forward'):
                Forward_Gluon += v[0]
            elif k.__contains__('Quark') and k.__contains__('Central'):
                Central_Quark += v[0]
            elif k.__contains__('Gluon') and k.__contains__('Central'):
                Central_Gluon += v[0]

        # p_H_Q = Forward_Quark / np.sum(Forward_Quark)
        # p_L_Q = Central_Quark / np.sum(Central_Quark)
        # p_H_G = Forward_Gluon / np.sum(Forward_Gluon)
        # p_L_G = Central_Gluon / np.sum(Central_Gluon)

        # p_L_Q[p_L_Q==0] = np.inf
        # p_L_G[p_L_G==0] = np.inf
        # if not np.isfinite(p_H_Q / p_L_Q).any() or not np.isfinite(p_H_G / p_L_G).any():
        #     raise Exception(f"The values in the factor is invalid! Please check!")
        if np.sum(Forward_Quark)==0 or np.sum(Central_Quark) == 0:
            quark_factor[str(l_pt_bin)] = np.ones_like(Forward_Quark)
        else: 
            p_H_Q = Forward_Quark / np.sum(Forward_Quark)
            p_L_Q = Central_Quark / np.sum(Central_Quark)
            quark_factor[str(l_pt_bin)] = safe_array_divide(p_H_Q, p_L_Q)   

        if np.sum(Forward_Gluon)==0 or np.sum(Central_Gluon) == 0:
            gluon_factor[str(l_pt_bin)] = np.ones_like(Forward_Gluon)
        else:    
            p_H_G = Forward_Gluon / np.sum(Forward_Gluon)
            p_L_G = Central_Gluon / np.sum(Central_Gluon)
            #quark_factor[str(l_pt_bin)] = safe_array_divide(p_H_Q, p_L_Q) 
            gluon_factor[str(l_pt_bin)] = safe_array_divide(p_H_G, p_L_G)

    return quark_factor, gluon_factor

def Initialize_Map():
    # 1200_LeadingJet_Forward_ntrk
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

def Minitree2Hist(input_folder, sum_of_weights, mc_syst_type, tree, reweight_var = None, reweight_factor = None, if_do_bdt=False, bdt_path = ''):
    if not reweight_var == None and reweight_factor == None:
        raise Exception(f"You specify reweight_type to be {reweight_var} but reweight_factor is not provided! ")

    if not reweight_var == None and not reweight_var in ["ntrk", "bdt", "newbdt"]:
        raise Exception(f"You specify reweight_type to be {reweight_var} but not supported! ")
         
    if if_do_bdt:
        if not bdt_path.exists():
            raise Exception(f"No bdt model found at path {bdt_path.as_posix()}")
        try:
            with open(bdt_path, 'rb') as f:
                bdt_model = pickle.load(f)
        except Exception as e:
            # everything else, possibly fatal
            print(traceback.format_exc(e))
            return

    HistMap = Initialize_Map()

    luminosity_periods = {
    "A" : 36000,
    "D" : 44500,
    "E" : 58500
    }

    xsec = np.array([7.8050E+07, 7.8050E+07, 2.4330E+06, 2.6450E+04, 2.5461E+02, 4.5532E+00, 2.5754E-01, 1.6215E-02, 6.2506E-04, 1.9639E-05])*1E3 # pb
    eff = np.array([9.753257E-01, 2.442497E-02, 9.863129E-03, 1.165838E-02, 1.336560E-02, 1.452648E-02, 9.471878E-03, 1.1097E-02, 1.015436E-02, 1.2056E-02])

    branch_names = ["pu_weight", "mconly_weight", "jet_fire", "jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]
    if mc_syst_type == "syst_ntrk_eff":
        branch_names[5] = "jet_nTracks_sys_eff"
    elif mc_syst_type == "syst_ntrk_fake":
        branch_names[5] = "jet_nTracks_sys_fake"

    branch_names_tobesaved = branch_names[3:]

    for i, dir in enumerate(input_folder):
        print(dir)

        JZ_slice_number = int(re.search("36470[0-9]", dir.stem).group()) % 100
        
        for file in sorted(dir.glob("*.minitrees.root")):
            print(f"        Doing on file {file.stem}, reweight var: {reweight_var}")
            branches_before_cut = uproot.open(file)[tree].arrays(branch_names, library='ak')
            branches_after_cut = Apply_Cuts(branches_before_cut = branches_before_cut, luminosity = luminosity_periods[period], 
                                            xsec = xsec, eff = eff, JZ_slice = JZ_slice_number, sum_of_weights = sum_of_weights)
            if len(branches_after_cut) == 0: # No jets passing the cuts, continue 
                continue 

            branches_to_be_saved = branches_after_cut[branch_names_tobesaved][:,:2]
            leading_jets = branches_to_be_saved[branch_names_tobesaved][:,0].to_numpy()
            subleading_jets = branches_to_be_saved[branch_names_tobesaved][:,1].to_numpy()
            leading_jets["jet_pt"] = leading_jets["jet_pt"]/1000
            subleading_jets["jet_pt"] = subleading_jets["jet_pt"]/1000

            leading_jets = np.dstack([leading_jets[field] for field in leading_jets.dtype.names])
            subleading_jets = np.dstack([subleading_jets[field] for field in subleading_jets.dtype.names])
            
            # leading_jets shape is (1, n_events, 7)
            jet_type_idx = 6 # Used in split_jet_type functions 

            #### new BDT prediction     
            if if_do_bdt:
                leading_jet_new_bdt = bdt_model.decision_function(leading_jets[0,:,:5])
                subleading_jet_new_bdt = bdt_model.decision_function(subleading_jets[0,:,:5])
                leading_jets = np.concatenate((leading_jets, np.broadcast_to(leading_jet_new_bdt[None,:,None], (leading_jets.shape[:2] + (1,)))), axis = 2)
                subleading_jets = np.concatenate((subleading_jets, np.broadcast_to(subleading_jet_new_bdt[None,:,None], (subleading_jets.shape[:2] + (1,)))), axis = 2)
                
                leading_jets[0,:,[6,7]] = leading_jets[0,:,[7,6]]
                subleading_jets[0,:,[6,7]] = subleading_jets[0,:,[7,6]]
                jet_type_idx = 7


            total_weight = branches_after_cut["total_weight"].to_numpy()
            leading_jets = np.concatenate((leading_jets, np.broadcast_to(total_weight[None, : , None], (leading_jets.shape[:2] + (1,)))), axis = 2)
            subleading_jets = np.concatenate((subleading_jets, np.broadcast_to(total_weight[None, : , None], (subleading_jets.shape[:2] + (1,)))), axis = 2)
            

            dijets = np.concatenate((leading_jets,subleading_jets), axis = 0) # the format is (2, n_events, 9) if no newbdt 
            dijets = np.swapaxes(dijets, 0, 1)# after swap, the format is (n_events, 2, 9)
            
            
            HistMap = Split_jets(HistMap = HistMap, dijets_array = dijets, reweight_var = reweight_var, reweight_factor = reweight_factor, jet_type_idx = jet_type_idx)

    return HistMap

           
def Apply_Cuts(branches_before_cut: ak.highlevel.Array, luminosity: int, xsec: List, eff: List, JZ_slice: int, sum_of_weights):

    total_weight_before_cut = luminosity * branches_before_cut["pu_weight"] * xsec[JZ_slice] * eff[JZ_slice] / sum_of_weights[JZ_slice - 1] # JZ_slice - 1 because of 1...9 -> 0...8
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
    branches = branches[np.abs(branches["total_weight"]) < 100] 
    # np.asarray(branches["total_weight"])[abnormal_weights] = 1 #

    return branches
    
def Split_jets(HistMap, dijets_array, reweight_var = None, reweight_factor = None, jet_type_idx=7):
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
            
            # assert label_eta[1] == "Central"
            if reweight_var in ["ntrk", "bdt", "newbdt"] and ki.__contains__(label_eta[1]): # label_eta[1] should be the Central 
                # Do reweighting here 
                if reweight_var == "ntrk":
                    reweight_var_idx = 2
                elif reweight_var == "bdt":
                    reweight_var_idx = 5
                elif reweight_var == "newbdt":
                    reweight_var_idx = 6
                
                reweight_var_bins = GetHistBin(reweight_var)

                reweighting_at_pt = reweight_factor[str(kj)]
                inds = np.digitize(splited_jet_pt_bin[:, reweight_var_idx], reweight_var_bins)
                inds[inds > (len(reweight_var_bins) - 1)] = len(reweight_var_bins) - 1  # Ensure that > the boundary are proper dealt
                # e.g. ntrk = 75 > 60 that is outside of reweighting_at_pt
                # reweight_var_bins[inds] will out of range for ntrk; just modify the inds

                splited_jet_pt_bin[:, -1] = splited_jet_pt_bin[:, -1] * reweighting_at_pt[inds-1]

                # inds_unique = np.unique(inds)

                # for unique_ind in inds_unique:
                #     np.where(splited_jet_pt_bin[:, reweight_var_idx] == unique_ind) 
            splited_pt_eta_jets_types = split_jet_type(splited_jet_pt_bin, jet_type_idx = jet_type_idx)

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

def split_jet_type(jets, jet_type_idx):
    gluon_idx = np.where(jets[:,jet_type_idx]==21)[0]
    light_quark_idx = np.where((jets[:,jet_type_idx]==1) | (jets[:,jet_type_idx]==2)| (jets[:,jet_type_idx]==3))[0]
    c_quark_idx = np.where(jets[:,jet_type_idx]==4)[0]
    b_quark_idx = np.where(jets[:,jet_type_idx]==5)[0]
    data_idx = np.where(jets[:,jet_type_idx]==-9999)[0]
    others_idx = np.where(jets[:,jet_type_idx]==-1)[0]

    all_list = [jets[gluon_idx], jets[light_quark_idx], jets[c_quark_idx], jets[b_quark_idx], jets[data_idx], jets[others_idx]]
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


def WriteHistRootFile(HistMap, output_file_name, TDirectory_name = "NoReighting"): 
    #### weird AssertionError
    print(f"Writing Histogram to the file: {output_file_name}")
    with uproot.update(output_file_name) as foutput:

        if TDirectory_name + ";1" in foutput.keys(recursive=False):
            print(f"{TDirectory_name} found in the root file. Trying to delete it.")
            del foutput[TDirectory_name]

            
    # foutput = uproot.update(output_file_name)
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
                inds = inds - 1 # This is important! 
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

def Make_Histogram_Pythia(sample_folder_path, period, sum_of_weights, output_path, mc_syst_type, tree, if_do_bdt, bdt_path):
    if not period in ["A", "D", "E"]:
        raise Exception(f'Period {period} not in supported periods. Currently supported: ["A", "D", "E"]')

    # period_JZslice = sorted(sample_folder_path.rglob(f"*pythia{period}*mc16_13TeV.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ*")) # For debug case, only a JZ slice is used
    period_JZslice = sorted(sample_folder_path.rglob(f"*mc16_13TeV.36470*.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ*")) # Read the whole period, can be A, D or E
    if len(period_JZslice)==0:
        raise Exception(f"No data files found! Please check the path {sample_folder_path.as_posix()}! ")
    output_file_name = output_path / f"dijet_pythia_mc16{period}"

    output_root_file =  output_file_name.as_posix() + ".root"
    output_pickle_file =  output_file_name.as_posix() + ".pkl"

    ntrk_quark_reweighting_file = output_file_name.as_posix() + "_ntrk_reweighting_quark_factor.pkl"
    ntrk_gluon_reweighting_file = output_file_name.as_posix() + "_ntrk_reweighting_gluon_factor.pkl"
    bdt_quark_reweighting_file = output_file_name.as_posix() + "_bdt_reweighting_quark_factor.pkl"
    bdt_gluon_reweighting_file = output_file_name.as_posix() + "_bdt_reweighting_gluon_factor.pkl"
    newbdt_quark_reweighting_file = output_file_name.as_posix() + "_newbdt_reweighting_quark_factor.pkl"
    newbdt_gluon_reweighting_file = output_file_name.as_posix() + "_newbdt_reweighting_gluon_factor.pkl"

    # HistMap = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = None, reweight_factor = None, mc_syst_type = mc_syst_type, tree = tree, if_do_bdt = if_do_bdt, bdt_path = bdt_path)  

    # uproot.recreate(output_root_file)

    # WriteHistRootFile(HistMap, output_root_file, TDirectory_name = "NoReweighting")
    # WritePickleFile(HistMap, output_pickle_file)  

    # Reweighting variants for ['ntrk', 'bdt'] * ['quark', 'gluon']
    # ntrk_reweighting_quark_factor, ntrk_reweighting_gluon_factor = Get_ReweightingFactor(input_file = output_root_file, reweight_var = "ntrk")
    # WritePickleFile(ntrk_reweighting_quark_factor, ntrk_quark_reweighting_file)
    # WritePickleFile(ntrk_reweighting_gluon_factor, ntrk_gluon_reweighting_file)

    # bdt_reweighting_quark_factor, bdt_reweighting_gluon_factor = Get_ReweightingFactor(input_file = output_root_file, reweight_var = "bdt")
    # WritePickleFile(bdt_reweighting_quark_factor, bdt_quark_reweighting_file)
    # WritePickleFile(bdt_reweighting_gluon_factor, bdt_gluon_reweighting_file)

    newbdt_reweighting_quark_factor, newbdt_reweighting_gluon_factor = Get_ReweightingFactor(input_file = output_root_file, reweight_var = "bdt")
    WritePickleFile(newbdt_reweighting_quark_factor, newbdt_quark_reweighting_file)
    WritePickleFile(newbdt_reweighting_gluon_factor, newbdt_gluon_reweighting_file)

    # HistMap_ntrk_reweighting_quark_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "ntrk", reweight_factor = ntrk_reweighting_quark_factor, mc_syst_type = mc_syst_type, tree = tree)
    # WriteHistRootFile(HistMap_ntrk_reweighting_quark_factor, output_root_file, TDirectory_name = "ntrk_Reweighting_Quark_Factor")

    # HistMap_ntrk_reweighting_gluon_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "ntrk", reweight_factor = ntrk_reweighting_gluon_factor, mc_syst_type = mc_syst_type, tree = tree)
    # WriteHistRootFile(HistMap_ntrk_reweighting_gluon_factor, output_root_file, TDirectory_name = "ntrk_Reweighting_Gluon_Factor")

    # HistMap_bdt_reweighting_quark_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "bdt", reweight_factor = bdt_reweighting_quark_factor, mc_syst_type = mc_syst_type, tree = tree)
    # WriteHistRootFile(HistMap_bdt_reweighting_quark_factor, output_root_file, TDirectory_name = "bdt_Reweighting_Quark_Factor")
    # #### Weird Assertation error below, don't know why 
    # HistMap_bdt_reweighting_gluon_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "bdt", reweight_factor = bdt_reweighting_gluon_factor, mc_syst_type = mc_syst_type, tree = tree)
    # WriteHistRootFile(HistMap_bdt_reweighting_gluon_factor, output_root_file, TDirectory_name = "bdt_Reweighting_Gluon_Factor")

    # The following are broken 
    # Not reading the newbdt response. Need to fix. TODO 
    HistMap_newbdt_reweighting_quark_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "newbdt", reweight_factor = newbdt_reweighting_quark_factor, mc_syst_type = mc_syst_type, tree = tree, if_do_bdt = if_do_bdt, bdt_path = bdt_path)
    WriteHistRootFile(HistMap_newbdt_reweighting_quark_factor, output_root_file, TDirectory_name = "newbdt_Reweighting_Quark_Factor")
    
    #### TODO FIX 
    #### Second time del failed 
    #### RuntimeError: segment of data to release overlaps one already marked as free: releasing [2244482, 2244643) but [2244381, 2244542) is free
    # HistMap_newbdt_reweighting_gluon_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "newbdt", reweight_factor = newbdt_reweighting_gluon_factor, mc_syst_type = mc_syst_type, tree = tree, if_do_bdt = if_do_bdt, bdt_path = bdt_path)
    # WriteHistRootFile(HistMap_newbdt_reweighting_gluon_factor, output_root_file, TDirectory_name = "newbdt_Reweighting_Quark_Factor")
    
    # HistMap_newbdt_reweighting_quark_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "newbdt", reweight_factor = newbdt_reweighting_quark_factor, mc_syst_type = mc_syst_type, tree = tree)
    # WriteHistRootFile(HistMap_newbdt_reweighting_quark_factor, output_root_file, TDirectory_name = "newbdt_Reweighting_Quark_Factor")
    
    # HistMap_newbdt_reweighting_gluon_factor = Minitree2Hist(input_folder = period_JZslice, sum_of_weights = sum_of_weights, reweight_var = "newbdt", reweight_factor = newbdt_reweighting_gluon_factor, mc_syst_type = mc_syst_type, tree = tree)
    # WriteHistRootFile(HistMap_newbdt_reweighting_gluon_factor, output_root_file, TDirectory_name = "newbdt_Reweighting_Gluon_Factor")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script converts MC minitrees to histogram in root and pkl. Reweighting is done.')
    parser.add_argument('--path', help='The path to the minitree files')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E'])
    parser.add_argument('--output-path', help='Output path')
    parser.add_argument('--do-syst', help='if we do the systmatics', action='store_true')
    parser.add_argument('--mc-syst-type', help='Systematics type', choices=['nominal', 'syst_ntrk_eff', 'syst_ntrk_fake'], default='nominal')
    parser.add_argument('--tree', help='Systematics trees', default='nonimal')
    parser.add_argument('--do-BDT', help='if use new BDT to do inference', action='store_true')
    parser.add_argument('--BDT-path', help='if use new BDT to do inference', default='/global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/models/bdt_model_flat_pt_gridsearchCV.model')
    args = parser.parse_args()

    minitrees_folder_path = Path(args.path)
    period = args.period
    output_path = Path(args.output_path)
    hist_folder_path = minitrees_folder_path.parent / (minitrees_folder_path.stem + "_hist")
    
    if_do_syst = args.do_syst
    if if_do_syst:
        mc_syst_type = args.mc_syst_type
        tree = args.tree
    else:
        mc_syst_type = "nominal"
        tree = "nominal"

    if_do_bdt = args.do_BDT
    bdt_path = Path(args.BDT_path)
 
    sum_of_weights_file_path = hist_folder_path / f"SumofWeights_mc16{period}.npy"

    # Check if the sumofweights is calculated 
    if sum_of_weights_file_path.exists():
        print(f"File {sum_of_weights_file_path} found, read from the npy file. ")
        sum_of_weights = np.load(sum_of_weights_file_path)
    else:
        print(f"File {sum_of_weights_file_path} not found, re-calculate it. ")
        sum_of_weights = Read_SumofWeights_Period(sample_folder_path = hist_folder_path, period = period)

    if if_do_syst:
        output_path = output_path / f"period{period}" / f"tree_{tree}" / mc_syst_type
    elif if_do_bdt:
        output_path = output_path / f"period{period}" / "newbdt"
    else:
        output_path = output_path / f"period{period}"

    if not (output_path).exists():
        output_path.mkdir(parents=True)

    Make_Histogram_Pythia(sample_folder_path=minitrees_folder_path, period=period, sum_of_weights=sum_of_weights, output_path = output_path, mc_syst_type = mc_syst_type, tree = tree, if_do_bdt = if_do_bdt, bdt_path = bdt_path)




