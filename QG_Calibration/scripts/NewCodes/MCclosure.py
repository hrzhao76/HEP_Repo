import argparse
from genericpath import exists
import uproot
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from pathlib import Path

def Construct(HistMap, n_bins):
    ## Construct data-like MC 
    Forward = np.zeros((n_bins))
    Central = np.zeros((n_bins))
    for k, v in HistMap.items():
        if k.__contains__('Forward'):
            Forward += v[0]
        elif k.__contains__('Central'):
            Central += v[0]

    ## Construct pure Quark vs Gluon 
    Quark = np.zeros((n_bins))
    Gluon = np.zeros((n_bins))
    for k, v in HistMap.items():
        if k.__contains__('Quark'):
            Quark += v[0]
        elif k.__contains__('Gluon'):
            Gluon += v[0]

    Forward_Quark = np.zeros((n_bins))
    Forward_Gluon = np.zeros((n_bins))
    Central_Quark = np.zeros((n_bins))
    Central_Gluon = np.zeros((n_bins))

    for k, v in HistMap.items():
        if k.__contains__('Quark') and k.__contains__('Forward'):
            Forward_Quark += v[0]
        elif k.__contains__('Gluon') and k.__contains__('Forward'):
            Forward_Gluon += v[0]
        elif k.__contains__('Quark') and k.__contains__('Central'):
            Central_Quark += v[0]
        elif k.__contains__('Gluon') and k.__contains__('Central'):
            Central_Gluon += v[0]
    return Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon 

def Calcu_Frac(Forward_Quark, Central_Quark, Forward, Central):
    try:
        frac_Forward_Quark = np.sum(Forward_Quark) / np.sum(Forward)
        frac_Central_Quark = np.sum(Central_Quark) / np.sum(Central)
    except RuntimeWarning:
        print(Forward)
        print(Central)

    frac_Forward_Gluon = 1 - frac_Forward_Quark
    frac_Central_Gluon = 1 - frac_Central_Quark

    f = np.array([[frac_Forward_Quark,  frac_Forward_Gluon], [frac_Central_Quark, frac_Central_Gluon]])

    return f, np.linalg.inv(f)

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

Map_var_title = {
    "pt": "$p_{T}$",
    "ntrk": "$N_{trk}$",
    "bdt": "BDT",
    "eta": "$\eta$",
    "c1": "$C_{1}$",
    "width": "W"
}
def Plot_ForwardCentral(pt, var, output_path, period, required_var, reweighting_option, p_Forward_Quark, p_Central_Quark, p_Forward_Gluon, p_Central_Gluon ):
    bin_edges = GetHistBin(var)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

    ax0.stairs(values = p_Forward_Quark, edges = bin_edges, color = 'blue', label = 'forward quark')
    ax0.stairs(values = p_Central_Quark, edges = bin_edges,  color = 'blue', linestyle='--', label = 'central quark' )

    ax0.stairs(values = p_Forward_Gluon, edges = bin_edges, color = 'red', label = 'forward gluon')
    ax0.stairs(values = p_Central_Gluon, edges = bin_edges, color = 'red', linestyle='--', label = 'central gluon' )

    ax0.set_ylabel("Normalized")
    ax0.legend()

    ax0.set_title(f"{pt} GeV: MC Q/G in eta region " + rf"{Map_var_title[var]}"  + f" distribution, {reweighting_option} reweighting")

    p_Central_Quark[p_Central_Quark == 0] = np.inf
    p_Central_Gluon[p_Central_Gluon == 0] = np.inf
    ax1.stairs(values = p_Forward_Quark/p_Central_Quark, edges = bin_edges, color = 'blue')
    ax1.stairs(values = p_Forward_Gluon/p_Central_Gluon, edges = bin_edges, color = 'red')
    ax1.set_ylabel("Forward/Central")
    ax1.set_ylim(0.7, 1.3)
    ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')


    ax1.plot()
    output_path_new = output_path / period / f"{required_var}_{reweighting_option}" / var 
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True, exist_ok =True)
    fig.savefig(output_path_new / f"MC_truth_Q_G_{pt}_{var}_{reweighting_option}_reweighting.jpg", dpi=300)
    plt.close()

def Plot_Extracted(pt, var, output_path, period, required_var, reweighting_option, p_Quark, extract_p_Quark, p_Gluon, extract_p_Gluon):
    bin_edges = GetHistBin(var)
        
    jet_types = ["quark", "gluon"]
    color_types = ["blue", "red"]
    plot_data = [[extract_p_Quark, p_Quark], [extract_p_Gluon, p_Gluon]]
    for i, jet_type in enumerate(jet_types): 
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
        ax0.stairs(values = plot_data[i][0], edges = bin_edges, color = color_types[i], label = f'{jet_type}, extracted MC')
        ax0.stairs(values = plot_data[i][1], edges = bin_edges, color = color_types[i], linestyle='--', label = f'{jet_type}, truth MC')
        ax0.legend()
        y_max = np.max([np.max(plot_data[i][0]), np.max(plot_data[i][1])])
        ax0.set_ylim(-0.01, y_max * 1.2)
        ax0.set_ylabel("Normalized")
        ax0.set_title(f"{pt} GeV {jet_type}: Extracted " + rf"{Map_var_title[var]}"  + f" distribution, {reweighting_option} reweighting")

        plot_data[i][1][plot_data[i][1]==0] = np.inf
        ax1.stairs(values = plot_data[i][0]/plot_data[i][1] , edges=bin_edges, color = color_types[i])
        ax1.set_ylim(0.7,1.3)
        ax1.set_ylabel("Extracted/Truth")
        ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')
        output_path_new = output_path / period / f"{required_var}_{reweighting_option}"  / var 
        if not output_path_new.exists():
            output_path_new.mkdir(parents = True)

        fig.savefig( output_path_new / f"MCclosure_extracted_{pt}_{jet_type}_{var}.jpg", dpi=300)
        plt.close()


def MCclosure(input_path, period, reweighting_option, output_path):
    # define some variables to make up the histogram names 
    label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]
    label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
    # label_var = ['bdt'] 
    label_pt = ["LeadingJet", "SubLeadingJet"]
    label_eta = ["Forward", "Central"]
    label_type = ["Gluon", "Quark", "B_Quark", "C_Quark"]

    # defile which TDirectory to look at based on {reweighting_option}
    reweighting_map = {
        "none" : "NoReweighting",
        "quark" : "Reweighting_Quark_Factor",
        "gluon" : "Reweighting_Gluon_Factor"
    }

    for required_var in ["ntrk", "bdt"]:
        if reweighting_option == "none":
            mcfile = uproot.open(input_path)[reweighting_map[reweighting_option]]
        else:
            mcfile = uproot.open(input_path)[f"{required_var}_" + reweighting_map[reweighting_option]]
        for var in label_var:
            for pt in label_pt_bin[:-1]:
                HistMap = {}

                for i, l_pt  in enumerate(label_pt):
                    for j, l_eta in enumerate(label_eta):
                        for k, l_type in enumerate(label_type):
                            key = str(pt) + "_" + l_pt + "_" + l_eta + "_" + l_type + "_" + var
                            HistMap[key] = mcfile[key].to_numpy()

                Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct(HistMap=HistMap, n_bins = len(GetHistBin(var)) - 1)

                f, f_inv = Calcu_Frac(Forward_Quark, Central_Quark, Forward, Central)

                # normalize 
                p_Quark = Quark / np.sum(Quark)
                p_Gluon = Gluon / np.sum(Gluon)

                p_Forward = Forward / np.sum(Forward)
                p_Central = Central / np.sum(Central)

                p_Forward_Quark = Forward_Quark / np.sum(Forward_Quark)
                p_Central_Quark = Central_Quark / np.sum(Central_Quark)
                p_Forward_Gluon = Forward_Gluon / np.sum(Forward_Gluon)
                p_Central_Gluon = Central_Gluon / np.sum(Central_Gluon)


                extract_p_Quark = f_inv[0][0] * p_Forward + f_inv[0][1]* p_Central 
                extract_p_Gluon = f_inv[1][0] * p_Forward + f_inv[1][1]* p_Central 

                Plot_ForwardCentral(pt = pt, var= var, output_path= output_path, 
                                    period= period, required_var = required_var,
                                    reweighting_option= reweighting_map[reweighting_option],
                                    p_Forward_Quark= p_Forward_Quark, p_Central_Quark= p_Central_Quark,
                                    p_Forward_Gluon= p_Forward_Gluon, p_Central_Gluon= p_Central_Gluon)

                Plot_Extracted(pt = pt, var= var, output_path= output_path, 
                                period= period, required_var = required_var,
                                reweighting_option= reweighting_map[reweighting_option],
                                p_Quark=p_Quark, extract_p_Quark = extract_p_Quark,
                                p_Gluon=p_Gluon, extract_p_Gluon = extract_p_Gluon)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script does the MC Closure test. ')
    parser.add_argument('--path', help='The path to the histogram file(.root file).')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E', "ADE"])
    parser.add_argument('--reweighting', help='The reweighting method', choices=['none', 'quark', 'gluon'])
    parser.add_argument('--output-path', help='Output path')
    args = parser.parse_args()

    root_file_path = Path(args.path)
    output_path = Path(args.output_path)
    period = args.period

    if root_file_path.suffix != ".root" :
        raise Exception(f"The input file {root_file_path} is not a root file! ")

    if period !=  root_file_path.stem[-len(period):]:
        raise Exception(f"The input file {root_file_path.stem} is not consistent with the period {period}!")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    MCclosure(input_path=root_file_path, period = period, reweighting_option = args.reweighting , output_path = output_path)
    

