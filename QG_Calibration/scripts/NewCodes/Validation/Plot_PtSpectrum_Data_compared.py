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
label_leadingtype = ["LeadingJet", "SubJet"]
label_etaregion = ["Forward", "Central"]
# label_jettype = ["Quark", "Gluon", "B_Quark", "C_Quark", "Other"]
label_jettype = ["Data"]
label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
n_bins_var = [60, 50, 60, 60, 60, 60]

def Read_Histogram_by_JetType(file, code_version="new"):
    Read_HistMap = {}
    Read_HistMap_Error = {}

    if code_version=="new":
        file = file["NoReweighting"]

    for i, jettype in enumerate(label_jettype):
        Read_HistMap[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        Read_HistMap_Error[jettype] = np.zeros((n_bins_var[0]), dtype=float)
        for pt in label_ptrange[:-1]:
            for leadingtype in label_leadingtype[0:1]:
                for eta_region in label_etaregion: 
                    Read_HistMap[jettype] += file[f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}"].to_numpy()[0]
                    Read_HistMap_Error[jettype] += file[f"{pt}_{leadingtype}_{eta_region}_{jettype}_{label_var[0]}_err"].to_numpy()[0]

    return Read_HistMap, Read_HistMap_Error

def Plot_PtSpectrum_Data(input_path, period, slice):
    output_path = input_path / "pt_spectrum"
    output_path.mkdir( exist_ok=True)
    newfile_path = input_path  / f"dijet_data_NEW_{period}{slice}.root"
    newfile = uproot.open(newfile_path)

    oldfile_path = input_path  / f"dijet_data_OLD_{period}{slice}.root"
    oldfile = uproot.open(oldfile_path)

    Read_HistMap_newfile, Read_HistMap_Error_newfile = Read_Histogram_by_JetType(newfile, code_version="new")
    Read_HistMap_oldfile, Read_HistMap_Error_oldfile = Read_Histogram_by_JetType(oldfile, code_version="old")


    fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
    custom_bins = np.linspace(0, 2000, 61)
    pt_bin_centers = 1/2 * (custom_bins[:-1] + custom_bins[1:])


    ax.fill_between(pt_bin_centers, 0, Read_HistMap_newfile[label_jettype[0]], label="New Code", step = 'mid')
    ax.scatter(pt_bin_centers, Read_HistMap_oldfile[label_jettype[0]], color= "black", marker= "o", label = "Old Code")


    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(Read_HistMap_newfile[label_jettype[0]], Read_HistMap_oldfile[label_jettype[0]])
        ratio[ratio == np.inf] = 0
        ratio = np.nan_to_num(ratio)

    ax1.stairs(values = ratio, edges=custom_bins, color = "black", linestyle=':', label = 'New / Old', baseline=None)
    ax1.hlines(y = 1, xmin = 500, xmax = 2000, color = 'gray', linestyle = '--')
    ax1.set_ylabel("Ratio")
    ax1.set_ylim(0.7, 1.3)
    ax.set_yscale('log')
    ax.set_xlim(500, 2000)
    ax.set_title( f'{period} {slice } ' + 'Leading Jet $p_{T}$ Spectrum Component')
    ax.set_xlabel('Jet $p_{\mathrm{T}}$ [GeV]')
    ax.set_ylabel('Number of Events')
    ax.legend()
    ax1.legend()
    fig.show()

    # assert np.sum(ratio[ratio!=0]) == len(ratio[ratio!=0])
    print(output_path.as_posix())
    fig.savefig(output_path.as_posix() + "/" + f"pt_spectrum_compared_{period}{slice}", dpi = 100)

if __name__ == '__main__':
    """This python script compares pt spectrum distributions of new codes and old codes. 
    The input path is hardcoded as /global/cfs/projectdirs/atlas/hrzhao/qgcal/New_Codes/Validation/CheckData{period}{slice}.
    """
    parser = argparse.ArgumentParser(description = 'This python script compares pt spectrum distributions of new codes and old codes.')
    parser.add_argument('--period', help='The Data period', choices=['15', '16', '17', '18'])
    parser.add_argument('--slice', help='slices')
    args = parser.parse_args()

    period = args.period
    slice = args.slice
    input_path = Path(f"/global/cfs/projectdirs/atlas/hrzhao/qgcal/New_Codes/Validation/CheckData{period}{slice}")
    Plot_PtSpectrum_Data(input_path = input_path, period = period, slice = slice)
