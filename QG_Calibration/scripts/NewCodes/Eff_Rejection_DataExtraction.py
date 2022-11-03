import argparse
from genericpath import exists
import uproot
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from pathlib import Path
from sklearn import metrics

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

def Construct_Data(HistMap_Data, n_bins):
    ## Construct data-like MC 
    Forward = np.zeros((n_bins))
    Central = np.zeros((n_bins))
    for k, v in HistMap_Data.items():
        if k.__contains__('Forward'):
            Forward += v[0]
        elif k.__contains__('Central'):
            Central += v[0]
    return Forward, Central

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


def Draw_ROC(input_mc_path, input_data_path, period, reweighting_option, output_path):
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

    WPs = [0.5, 0.6, 0.7, 0.8]
    # for required_var in ["ntrk", "bdt"]:
    for WP in WPs:
        for required_var in ["ntrk", "bdt"]:
            if reweighting_option == "none":
                mcfile = uproot.open(input_mc_path)[reweighting_map[reweighting_option]]
                datafile = uproot.open(input_data_path)[reweighting_map[reweighting_option]]
            else:
                mcfile = uproot.open(input_mc_path)[f"{required_var}_" + reweighting_map[reweighting_option]]
                datafile = uproot.open(input_data_path)[f"{required_var}_" + reweighting_map[reweighting_option]]

            quark_effs_at_pt = np.zeros(len(label_pt_bin)-1)
            gluon_rejs_at_pt = np.zeros(len(label_pt_bin)-1)
            quark_effs_data_at_pt = np.zeros(len(label_pt_bin)-1)
            gluon_rejs_data_at_pt = np.zeros(len(label_pt_bin)-1)

            for ii, pt in enumerate(label_pt_bin[:-1]):

                HistMap = {}
                HistMap_Data = {}
                for i, l_pt  in enumerate(label_pt):
                    for j, l_eta in enumerate(label_eta):
                        key_data = str(pt) + "_" + l_pt + "_" + l_eta + "_" + "Data" + "_" + required_var
                        HistMap_Data[key_data] = datafile[key_data].to_numpy()
                        for k, l_type in enumerate(label_type):
                            key = str(pt) + "_" + l_pt + "_" + l_eta + "_" + l_type + "_" + required_var
                            HistMap[key] = mcfile[key].to_numpy()

                Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct(HistMap=HistMap, n_bins = len(GetHistBin(required_var)) - 1)
                Forward_Data, Central_Data  = Construct_Data(HistMap_Data=HistMap_Data, n_bins = len(GetHistBin(required_var)) - 1)

                f, f_inv = Calcu_Frac(Forward_Quark, Central_Quark, Forward, Central)

                # normalize 
                p_Quark = Quark / np.sum(Quark)
                p_Gluon = Gluon / np.sum(Gluon)

                p_Forward = Forward / np.sum(Forward)
                p_Central = Central / np.sum(Central)
                p_Forward_Data = Forward_Data / np.sum(Forward_Data)
                p_Central_Data = Central_Data / np.sum(Central_Data)

                p_Forward_Quark = Forward_Quark / np.sum(Forward_Quark)
                p_Central_Quark = Central_Quark / np.sum(Central_Quark)
                p_Forward_Gluon = Forward_Gluon / np.sum(Forward_Gluon)
                p_Central_Gluon = Central_Gluon / np.sum(Central_Gluon)


                extract_p_Quark = f_inv[0][0] * p_Forward + f_inv[0][1]* p_Central 
                extract_p_Gluon = f_inv[1][0] * p_Forward + f_inv[1][1]* p_Central 
                extract_p_Quark_Data = f_inv[0][0] * p_Forward_Data + f_inv[0][1]* p_Central_Data 
                extract_p_Gluon_Data = f_inv[1][0] * p_Forward_Data + f_inv[1][1]* p_Central_Data 

                extract_p_Quark_cum_sum = np.cumsum(extract_p_Quark)
                cut = np.where(extract_p_Quark_cum_sum >= WP)[0][0] + 1
                quark_effs_at_pt[ii] = np.sum(extract_p_Quark[:cut])
                gluon_rejs_at_pt[ii] = np.sum(extract_p_Gluon[cut:])

                quark_effs_data_at_pt[ii] = np.sum(extract_p_Quark_Data[:cut])
                gluon_rejs_data_at_pt[ii] = np.sum(extract_p_Gluon_Data[cut:])
                

                
            fig, ax0 = plt.subplots()
            ax0.stairs(values = quark_effs_at_pt, edges = label_pt_bin, color = "blue", label = 'Quark Efficiency, Extracted MC', baseline=None)
            ax0.stairs(values = gluon_rejs_at_pt, edges = label_pt_bin, color = "red", label = 'Gluon Rejection, Extracted MC', baseline=None)
            
            ax0.stairs(values = quark_effs_data_at_pt, edges = label_pt_bin, color = "blue", linestyle=':', label = 'Quark Efficiency, Extracted Data', baseline=None)
            ax0.stairs(values = gluon_rejs_data_at_pt, edges = label_pt_bin, color = "red", linestyle=':', label = 'Gluon Rejection, Extracted Data', baseline=None)
            
            ax0.legend()
            ax0.set_yticks(np.linspace(0, 1, 21))
            ax0.set_ylim(0.3, 1.2)
            ax0.grid()
            ax0.set_title(f"{required_var} for extracted q/g at {WP} WP")
            output_path_new = output_path / period / "WPs" / f"{required_var}_{reweighting_map[reweighting_option]}" 
            if not output_path_new.exists():
                output_path_new.mkdir(parents = True)
            fig.savefig( output_path_new/ f"{required_var}_WP_{WP}.jpg", dpi=300)
            plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script draw the ROC curves. ')
    parser.add_argument('--path-mc', help='The path to the MC histogram file(.root file).')
    parser.add_argument('--path-data', help='The path to the Data histogram file(.root file).')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E', "ADE"])
    parser.add_argument('--reweighting', help='The reweighting method', choices=['none', 'quark', 'gluon'])
    parser.add_argument('--output-path', help='Output path')
    args = parser.parse_args()

    mc_file_path = Path(args.path_mc)
    data_file_path = Path(args.path_data)
    output_path = Path(args.output_path)
    period = args.period

    if mc_file_path.suffix != ".root" :
        raise Exception(f"The input file {mc_file_path} is not a root file! ")

    if period !=  mc_file_path.stem[-len(period):]:
        raise Exception(f"The input file {mc_file_path.stem} is not consistent with the period {period}!")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    Draw_ROC(input_mc_path=mc_file_path, input_data_path=data_file_path, period = period, reweighting_option = args.reweighting , output_path = output_path)
    

