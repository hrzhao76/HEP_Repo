import argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot 
import awkward as ak
from pathlib import Path
from typing import Dict, List 
import re
import pickle

label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
label_leadingtype = ["LeadingJet", "SubLeadingJet"]
label_etaregion = ["Forward", "Central"]
label_jettype = ["Quark", "Gluon", "B_Quark", "C_Quark", "Other"]
label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
n_bins_var = [60, 50, 60, 60, 60, 60]

def Read_Histogram_by_JetType(file, l_leadingtype, sampletype="MC", code_version="new", reweighting_option = "NoReweighting"):
    Read_HistMap = {}
    Read_HistMap_Error = {}

    if code_version=="new":
        file = file[reweighting_option]

    if sampletype== "MC":
        label_jettype = ["Quark", "Gluon", "B_Quark", "C_Quark", "Other"]
    elif sampletype == "Data":
        label_jettype = ["Data"]
    
    if l_leadingtype== "leading":
        i_leading = 0
    elif l_leadingtype == "subleading":
        i_leading = 1
    else:
        raise Exception(f"not support leading type{l_leadingtype}")
    for i, jettype in enumerate(label_jettype):
        Read_HistMap[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        Read_HistMap_Error[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        for pt in label_ptrange[:-1]:
            for leadingtype in label_leadingtype[i_leading:i_leading+1]:
                for eta_region in label_etaregion: 
                    Read_HistMap[jettype] += file[f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}"].to_numpy()[0]
                    Read_HistMap_Error[jettype] += file[f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}_err"].to_numpy()[0]

    return Read_HistMap, Read_HistMap_Error


def cal_sum(Read_HistMap, Read_Hist_Error):
    """For MC sample only, this func is to calculate the sum of each time. 

    Args:
        Read_HistMap (Dict): the output of Read_Histogram by JetType

    Returns:
        np.array: sum of different types 
    """
    MC_jet_types = ['C_Quark', 'B_Quark', 'Gluon', 'Quark']

    MC_jets = []
    MC_jets_err = []
    for MC_jet_type in MC_jet_types:
        MC_jets.append(Read_HistMap[MC_jet_type])
        MC_jets_err.append(Read_Hist_Error[MC_jet_type])

    MC_jets = np.array(MC_jets)
    MC_jets_err = np.array(MC_jets_err)

    cumsum_MC_jets = np.cumsum(MC_jets, axis = 0)
    cumsum_MC_jets = np.concatenate((np.zeros((n_bins_var[0]), dtype=float)[None,:], cumsum_MC_jets))

    cumsum_MC_jets_error = np.cumsum(MC_jets_err, axis = 0)
    cumsum_MC_jets_error = np.concatenate((np.zeros((n_bins_var[0]), dtype=float)[None,:], cumsum_MC_jets_error))

    assert np.allclose(Read_Hist_Error[MC_jet_types[0]], cumsum_MC_jets_error[1])
    return cumsum_MC_jets, cumsum_MC_jets_error

def safe_array_divide(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(numerator, denominator)
        ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
    return ratio

def plot_pt(Read_HistMap_MC,Read_HistMap_Error_MC, Read_HistMap_Data, Read_HistMap_Error_Data, l_leading_type, period, output_path):
    MC_jet_types = ['C_Quark', 'B_Quark', 'Gluon', 'Quark']
    cumsum_MC_jets, cumsum_MC_jets_err = cal_sum(Read_HistMap=Read_HistMap_MC, Read_Hist_Error = Read_HistMap_Error_MC)

    fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
    custom_bins = np.linspace(0, 2000, 61)
    pt_bin_centers = 1/2 * (custom_bins[:-1] + custom_bins[1:])

    for i in range(0, len(cumsum_MC_jets)-1):
        ax.fill_between(pt_bin_centers, cumsum_MC_jets[i], cumsum_MC_jets[i+1], label = MC_jet_types[i]+ f", num:{np.sum(Read_HistMap_MC[MC_jet_types[i]]):.2e}", step='mid')

    total_jet_MC = cumsum_MC_jets[-1]
    total_jet_Data = Read_HistMap_Data['Data']
    total_jet_error_MC = np.sqrt(cumsum_MC_jets_err[-1])
    total_jet_error_Data = np.sqrt(Read_HistMap_Error_Data["Data"])

    # ax.stairs(values=cumsum_MC_jets[-1], edges=custom_bins, label = "Total MC"+ f"num. {np.sum(cumsum_MC_jets[-1]):.2f}" )
    ax.errorbar(x = pt_bin_centers, y = total_jet_MC, yerr = total_jet_error_MC, drawstyle = 'steps-mid', label = "Total MC"+ f", num:{np.sum(cumsum_MC_jets[-1]):.2e}")
    ax.errorbar(x = pt_bin_centers, y = total_jet_Data, yerr = total_jet_error_Data, drawstyle = 'steps-mid', color= "black", linestyle='', marker= "o", label = "Data" + f", num:{np.sum(Read_HistMap_Data['Data']):.2e}")


    with np.errstate(divide='ignore', invalid='ignore'):
        ratio1 = np.true_divide(np.sqrt(cumsum_MC_jets_err[-1]), cumsum_MC_jets[-1])
        ratio2 = np.true_divide(np.sqrt(Read_HistMap_Error_Data["Data"]), Read_HistMap_Data['Data'])
        for ratio in ratio1, ratio2:
            ratio[ratio == np.inf] = 0
            ratio = np.nan_to_num(ratio)

    ratio = safe_array_divide(total_jet_MC, total_jet_Data)
    error_ratio1 = safe_array_divide(total_jet_error_MC, total_jet_MC)
    error_ratio2 = safe_array_divide(total_jet_error_Data, total_jet_Data)
    error_ratio = np.sqrt((error_ratio1)**2 + (error_ratio2)**2) * ratio

    ax1.errorbar(pt_bin_centers, ratio, yerr = error_ratio, color= "black", drawstyle = 'steps-mid', label = 'MC / Data')
    # ax1.stairs(values = ratio, edges=custom_bins, color = "black", linestyle=':', label = 'MC / Data', baseline=None)
    ax1.hlines(y = 1, xmin = 500, xmax = 2000, color = 'gray', linestyle = '--')
    ax1.set_ylabel("Ratio")
    ax1.set_ylim(0.7, 1.3)
    ax.set_yscale('log')
    ax.set_xlim(500, 2000)
    ax.set_title(f'MC16{period} {l_leading_type}' +  ' Jet $p_{T}$ Spectrum Component')
    ax.set_xlabel('Jet $p_{\mathrm{T}}$ [GeV]')
    ax.set_ylabel('Number of Events')
    ax.legend()
    ax1.legend()

    fig.savefig(output_path/f'pt_MC16{period}_{l_leading_type}')

def Plot_Spectrum(mcfile_path, datafile_path, period, output_path):
    # mcfile_path = "/global/cfs/projectdirs/atlas/hrzhao/qgcal/Processed_Samples/dijet_pythia_mc16A.root"
    mcfile = uproot.open(mcfile_path)
    # datafile_path = "/global/cfs/projectdirs/atlas/hrzhao/qgcal/Processed_Samples_Data/data1516/dijet_data_1516.root"
    datafile = uproot.open(datafile_path)

    for var_leading in ["leading", "subleading"]:
        Read_HistMap_MC, Read_HistMap_Error_MC = Read_Histogram_by_JetType(mcfile, l_leadingtype=var_leading, code_version="new")
        Read_HistMap_Data, Read_HistMap_Error_Data = Read_Histogram_by_JetType(datafile, l_leadingtype=var_leading, sampletype = "Data", code_version="new")

        plot_pt(Read_HistMap_MC = Read_HistMap_MC, Read_HistMap_Error_MC = Read_HistMap_Error_MC,
        Read_HistMap_Data = Read_HistMap_Data, Read_HistMap_Error_Data = Read_HistMap_Error_Data,
        l_leading_type=var_leading, period=period, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script plot jet pt spectrum in 1500-2000 GeV')
    parser.add_argument('--mcpath', help='The path to the MC root file')
    parser.add_argument('--datapath', help='The path to the data root files')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E', 'ADE'])
    parser.add_argument('--output-path', help='Output path')
    args = parser.parse_args()

    mcfile_path = Path(args.mcpath)
    datafile_path = Path(args.datapath)
    period = args.period
    output_path = Path(args.output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    Plot_Spectrum(mcfile_path= mcfile_path, datafile_path = datafile_path, period=period, output_path = output_path)
