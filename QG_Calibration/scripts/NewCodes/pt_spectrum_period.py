# %%
import argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot 
import awkward as ak
from pathlib import Path
from typing import Dict, List 
import re
import pickle

def Plot_Spectrum(mcfile_path, datafile_path, period, output_path):
    # mcfile_path = "/global/cfs/projectdirs/atlas/hrzhao/qgcal/Processed_Samples/dijet_pythia_mc16A.root"
    mcfile = uproot.open(mcfile_path)
    # datafile_path = "/global/cfs/projectdirs/atlas/hrzhao/qgcal/Processed_Samples_Data/data1516/dijet_data_1516.root"
    datafile = uproot.open(datafile_path)

    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    label_leadingtype = ["LeadingJet", "SubJet"]
    label_etaregion = ["Forward", "Central"]
    label_jettype = ["Quark", "Gluon", "B_Quark", "C_Quark", "Other"]
    label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
    n_bins_var = [60, 50, 60, 60, 60, 60]

    Read_HistMap = {}
    Read_HistMap_Error = {}

    for i, jettype in enumerate(label_jettype):
        Read_HistMap[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        Read_HistMap_Error[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        for pt in label_ptrange[:-1]:
            for leadingtype in label_leadingtype[0:1]:
                for eta_region in label_etaregion: 
                    Read_HistMap[jettype] += mcfile["NoReweighting"][f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}"].to_numpy()[0]
                    Read_HistMap_Error[jettype] += mcfile["NoReweighting"][f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}_err"].to_numpy()[0]

    Read_HistMap_Data = {}
    Read_HistMap_Error_Data = {}
    label_jettype_data = ["Data"]
    for i, jettype in enumerate(label_jettype_data):
        Read_HistMap_Data[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        Read_HistMap_Error_Data[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        for pt in label_ptrange[:-1]:
            for leadingtype in label_leadingtype[0:1]:
                for eta_region in label_etaregion: 
                    Read_HistMap_Data[jettype] += datafile["NoReweighting"][f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}"].to_numpy()[0]
                    Read_HistMap_Error_Data[jettype] += datafile["NoReweighting"][f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}_err"].to_numpy()[0]

    MC_jet_types = [*Read_HistMap.keys()]
    MC_jet_types.reverse()

    total_error = np.zeros(60, dtype=np.float32)
    for mc_type in MC_jet_types[1:]:
        total_error += Read_HistMap_Error[mc_type]

    fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
    custom_bins = np.linspace(0, 2000, 61)
    pt_bin_centers = 1/2 * (custom_bins[:-1] + custom_bins[1:])

    ax.fill_between(pt_bin_centers, 0, Read_HistMap[MC_jet_types[1]], label=  MC_jet_types[1], step = 'mid')
    cum_sum = Read_HistMap[MC_jet_types[1]]
    # # for i in range(1, len(MC_jet_types) - 1):
    for i in range(1, len(MC_jet_types) - 1):
        cum_sum_addone = cum_sum + Read_HistMap[MC_jet_types[i+1]]
        ax.fill_between(pt_bin_centers, cum_sum, cum_sum_addone, label = MC_jet_types[i+1], step='mid')
        cum_sum = cum_sum_addone
    # ax.fill_between(pt_bin_centers, Read_HistMap, data=["C_Quark", "B_Quark", "Gluon", "Quark"], step = 'mid')

    # ax.hist(total, bins =custom_bins,  label = 'Total MC')
    ax.errorbar(pt_bin_centers, cum_sum_addone, yerr= np.sqrt(total_error), marker = "o", color = "black", linestyle='', label = "Total MC")
    # ax.scatter(pt_bin_centers, Read_HistMap_Data['Data'], color= "purple", marker= "^", label = "Data")
    ax.errorbar(pt_bin_centers, Read_HistMap_Data['Data'], yerr= np.sqrt(Read_HistMap_Error_Data["Data"]) , color= "purple", linestyle='', marker= "^", label = "Data")




    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(cum_sum_addone,Read_HistMap_Data['Data'])
        ratio[ratio == np.inf] = 0
        ratio = np.nan_to_num(ratio)

    ax1.stairs(values = ratio, edges=custom_bins, color = "black", linestyle=':', label = 'MC / Data', baseline=None)
    ax1.hlines(y = 1, xmin = 500, xmax = 2000, color = 'gray', linestyle = '--')
    ax1.set_ylabel("Ratio")
    ax1.set_ylim(0.7, 1.3)
    ax.set_yscale('log')
    ax.set_xlim(500, 2000)
    ax.set_title(f'MC16{period} ' + r'Leading Jet $p_{T}$ Spectrum Component')
    ax.set_xlabel('Jet $p_{\mathrm{T}}$ [GeV]')
    ax.set_ylabel('Number of Events')
    ax.legend()
    ax1.legend()
    fig.show()

    fig.savefig(output_path/f'pt_MC16{period}')


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
