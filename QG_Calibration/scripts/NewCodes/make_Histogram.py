import argparse
import numpy as np

import uproot 
import awkward as ak
from pathlib import Path

from read_SumofWeights import Read_SumofWeights_Period
from typing import Dict, List 
import re

label_pt = ["LeadingJet", "SubJet"]
label_eta = ["Forward", "Central"]
label_type = ["Gluon", "Quark", "C_Quark", "B_Quark", "Data", "Other"]
label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]

def Apply_Cuts(branches_before_cut: ak.highlevel.Array, luminosity: int, xsec: List, eff: List, JZ_slice: int, sum_of_weights):

    total_weight_before_cut = luminosity * branches_before_cut["pu_weight"] * xsec[JZ_slice] * eff[JZ_slice] / sum_of_weights[JZ_slice - 1] # JZ_slice - 1 because of 1...9 -> 0...8
    branches = ak.with_field(base = branches_before_cut, what = total_weight_before_cut, where = "total_weight")

    # event trigger selection 
    event_trigger_idx = branches["jet_fire"] == 1
    branches = branches[event_trigger_idx]

    # pT cut 
    pt_cut_idx = branches["jet_pt"][:,0] > 500000
    branches = branches[pt_cut_idx]

    pt_max_cut_idx = branches["jet_pt"][:,0] < 2000000 
    branches = branches[pt_max_cut_idx]

    # ratio < 1.5
    branches = branches[branches["jet_pt"][:,0]/branches["jet_pt"][:,1] < 1.5]

    # eta cut 
    branches = branches[np.abs(branches["jet_eta"][:,0]) < 2.1]
    branches = branches[np.abs(branches["jet_eta"][:,1]) < 2.1]


    # weight cut, for weight > 100 set it to 1 
    abnormal_weights = np.abs(branches["total_weight"]) > 100
    np.asarray(branches["total_weight"])[abnormal_weights] = 1

    return branches
    
def Split_jets(HistMap, dijets_array):
    splited_pt_eta_jets = split_pt_eta_jet(dijets_array)
    label_pt_eta = [label1_pt + "_" + label2_eta for label1_pt in label_pt for label2_eta in label_eta]

    n_jets = 0
    for i, splited_pt_eta_jet in enumerate(splited_pt_eta_jets):
        n_jets += len(splited_pt_eta_jet)

        splited_pt_eta_jets_types = split_jet_type(splited_pt_eta_jet)

        for j, jet_type in enumerate(splited_pt_eta_jets_types):
            if jet_type.shape[0] == 0:
                continue 

            splited_jet_pt_bins = split_jet_pt(jet_type)

            for k, splited_jet_pt_bin in splited_jet_pt_bins.items():
                prefix = str(k) + "_" + label_pt_eta[i] + "_" + label_type[j]
                FillHisto(HistMap, prefix, splited_jet_pt_bin)

    assert n_jets == len(dijets_array)*2 # Check if jets are splited correctly 





def split_pt_eta_jet(jets):
    # divide jets into 4 regions 
    forward_idx = np.argmax(np.abs(jets[:,:,1]), axis=1) # compare abs eta of jets inside events
    central_idx = -1*forward_idx+1

    leading_forward_idx = forward_idx == 0 # leading forward 
    subleading_forward_idx = forward_idx == 1 # subleading forward 

    leading_central_idx = central_idx == 0 # leading central 
    subleading_central_idx = central_idx == 1 # subleading central 

    return [jets[leading_forward_idx, 0, :], jets[leading_central_idx, 0, :],  
            jets[subleading_forward_idx, 1, :], jets[subleading_central_idx, 1, :]]

def split_jet_type(jets):
    gluon_idx = np.where(jets[:,6]==21)[0]
    light_quark_idx = np.where((jets[:,6]==1) | (jets[:,6]==2)| (jets[:,6]==3))[0]
    c_quark_idx = np.where(jets[:,6]==4)[0]
    b_quark_idx = np.where(jets[:,6]==5)[0]
    data_idx = np.where(jets[:,6]==-9999)[0]
    others_idx = np.where(jets[:,6]==-1)[0]

    gluon = jets[gluon_idx]
    quark = jets[light_quark_idx]
    c_quark = jets[c_quark_idx]
    b_quark = jets[b_quark_idx]
    data = jets[data_idx]
    others = jets[others_idx]

    return [gluon, quark, c_quark, b_quark, data, others]

def split_jet_pt(jets):
    splited_jet_pt = {}
    for i, pt_start in enumerate(label_pt_bin[:-1]):
        pt_selected_idx = np.where((jets[:,0] >= pt_start) & (jets[:,0] < label_pt_bin[i+1]))[0]
        splited_jet_pt[pt_start] = jets[pt_selected_idx]

    return splited_jet_pt


###### define functions
def GetHistBin(histname):
	if 'pt' in histname:
		return 60,0,2000
	elif 'eta' in histname:
		return 50,-2.5,2.5
	elif 'ntrk' in histname:
		return 60,0,60
	elif 'bdt' in histname:
		return 60,-0.8,0.7
	elif 'width' in histname:
		return 60,0.,0.4
	elif 'c1' in histname:
		return 60,0.,0.4
	elif 'newBDT' in histname:
		return 60,-0.8,0.7

def FillTH1F(HistMap, histname, var, w,):
    if 'Data' in histname:
        w = np.ones(len(var))
    if histname in HistMap:
        HistMap[histname][0].append(var)
        HistMap[histname][1].append(w)
    else:
        HistMap[histname] = [[],[]] #The first list is for the data, the second for the weights
        HistMap[histname][0].append(var)
        HistMap[histname][1].append(w)

def FillHisto(HistMap, prefix, jetlist):
	for i in range(6):
		FillTH1F(HistMap, prefix+"_"+label_var[i], jetlist[:,i], jetlist[:,7])

	# FillTH1F(HistMap, prefix+"_"+label_var[6], jetlist[:,8], jetlist[:,7]) # Fill new BDT values 




def Make_Histogram(sample_folder_path, period, sum_of_weights):
    if not period in ["A", "D", "E"]:
        raise Exception(f'Period {period} not in supported periods. Currently supported: ["A", "D", "E"]')

    luminosity_periods = {
    "A" : 36000,
    "D" : 44500,
    "E" : 58500
    }

    xsec = np.array([7.8050E+07, 7.8050E+07, 2.4330E+06, 2.6450E+04, 2.5461E+02, 4.5532E+00, 2.5754E-01, 1.6215E-02, 6.2506E-04, 1.9639E-05])*1E3 # pb
    eff = np.array([9.753257E-01, 2.442497E-02, 9.863129E-03, 1.165838E-02, 1.336560E-02, 1.452648E-02, 9.471878E-03, 1.1097E-02, 1.015436E-02, 1.2056E-02])

    branch_names = ["pu_weight", "mconly_weight", "jet_fire", "jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]
    branch_names_tobesaved = ["jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]
    period_JZslice = sorted(sample_folder_path.rglob(f"*pythia{period}*mc16_13TeV.36470*.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ*"))

    HistMap = {}

    for i, dir in enumerate(period_JZslice):
        print(dir)

        JZ_slice_number = int(re.search("36470[0-9]", dir.stem).group()) % 100
        
        for file in sorted(dir.glob("*.minitrees.root")):
            print(f"        Doing on file {file.stem}")
            branches_before_cut = uproot.open(file)["nominal"].arrays(branch_names, library='ak')
            branches_after_cut = Apply_Cuts(branches_before_cut = branches_before_cut, luminosity = luminosity_periods[period], 
                                            xsec = xsec, eff = eff, JZ_slice = JZ_slice_number, sum_of_weights = sum_of_weights)
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
            dijets = np.swapaxes(dijets, 0, 1)

            Split_jets(HistMap = HistMap, dijets_array = dijets)

    foutput = uproot.recreate(f"./dijet_pythia_mc16{period}.root")

    for hist in HistMap.keys():
        nbin,binmin,binmax = GetHistBin(hist)
        if len(HistMap[hist][0]) == 0:
            continue
        all_values = HistMap[hist][0][0]
        all_weights = HistMap[hist][1][0]
        for i in range(1, len(HistMap[hist][0])):
            all_values = np.append(all_values, HistMap[hist][0][i])
            all_weights = np.append(all_weights, HistMap[hist][1][i])
        histogram = np.histogram(a = all_values, weights = all_weights, bins = nbin, range = (binmin,binmax))

        foutput[f"NoReweighitng/{hist}"] = histogram

        # weight = np.array(HistMap[hist][1])
        # binning = np.linspace(binmin,binmax,nbin)
        # sum_w2 = np.zeros([nbin], dtype=np.float32)
        # digits = np.digitize(HistMap[hist][0],binning)
        # for i in range(nbin):
        #     weights_in_current_bin = weight[0][np.where(digits == i)[0]]
        #     sum_w2[i] = np.sum(np.power(weights_in_current_bin, 2))
        # #print(sum_w2)
        # histogram_err = np.histogram(a = binning, weights = sum_w2, bins = nbin, range = (binmin,binmax))
        # foutput[hist+"_err"] = histogram_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script calculate pythia weights.')
    parser.add_argument('--path', help='The path to the hist files')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E'])
    args = parser.parse_args()

    minitrees_folder_path = Path(args.path)
    period = args.period
    hist_folder_path = minitrees_folder_path.parent / (minitrees_folder_path.stem + "_hist")


    sum_of_weights_file_path = hist_folder_path / f"SumofWeights_mc16{period}.npy"

    # Check if the sumofweights is calculated 
    if sum_of_weights_file_path.exists():
        print(f"File {sum_of_weights_file_path} found, read from the npy file. ")
        sum_of_weights = np.load(sum_of_weights_file_path)
    else:
        print(f"File {sum_of_weights_file_path} not found, re-calculate it. ")
        sum_of_weights = Read_SumofWeights_Period(sample_folder_path = hist_folder_path, period = period)

    Make_Histogram(sample_folder_path=minitrees_folder_path, period=period, sum_of_weights=sum_of_weights)
