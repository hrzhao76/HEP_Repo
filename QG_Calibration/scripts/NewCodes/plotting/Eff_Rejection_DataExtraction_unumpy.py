import argparse
from genericpath import exists
import uproot
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from pathlib import Path

from uncertainties import ufloat, unumpy

def Construct(HistMap, HistMap_Error, n_bins, sampletype):
    ## Construct data-like MC 
    Forward = np.zeros((n_bins))
    Central = np.zeros((n_bins))
    Forward_Error = np.zeros((n_bins))
    Central_Error = np.zeros((n_bins))

    for k, v in HistMap.items():
        if k.__contains__('Forward'):
            Forward += v
        elif k.__contains__('Central'):
            Central += v

    if sampletype == "Data":
        return Forward, Central

    ## Construct pure Quark vs Gluon 
    Quark = np.zeros((n_bins))
    Gluon = np.zeros((n_bins))
    for k, v in HistMap.items():
        if k.__contains__('Quark'):
            Quark += v
        elif k.__contains__('Gluon'):
            Gluon += v

    Forward_Quark = np.zeros((n_bins))
    Forward_Gluon = np.zeros((n_bins))
    Central_Quark = np.zeros((n_bins))
    Central_Gluon = np.zeros((n_bins))

    for k, v in HistMap.items():
        if k.__contains__('Quark') and k.__contains__('Forward'):
            Forward_Quark += v
        elif k.__contains__('Gluon') and k.__contains__('Forward'):
            Forward_Gluon += v
        elif k.__contains__('Quark') and k.__contains__('Central'):
            Central_Quark += v
        elif k.__contains__('Gluon') and k.__contains__('Central'):
            Central_Gluon += v
    return Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon 
    
def Construct_unumpy(HistMap_unumpy, n_bins, sampletype):
    ## Construct data-like MC 
    Forward_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Central_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 

    for k, v in HistMap_unumpy.items():
        if k.__contains__('Forward'):
            Forward_unumpy += v
        elif k.__contains__('Central'):
            Central_unumpy += v

    if sampletype == "Data":
        return Forward_unumpy, Central_unumpy

    ## Construct pure Quark vs Gluon 
    Quark_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Gluon_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    for k, v in HistMap_unumpy.items():
        if k.__contains__('Quark'):
            Quark_unumpy += v
        elif k.__contains__('Gluon'):
            Gluon_unumpy += v

    Forward_Quark_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Forward_Gluon_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Central_Quark_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Central_Gluon_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 

    for k, v in HistMap_unumpy.items():
        if k.__contains__('Quark') and k.__contains__('Forward'):
            Forward_Quark_unumpy += v
        elif k.__contains__('Gluon') and k.__contains__('Forward'):
            Forward_Gluon_unumpy += v
        elif k.__contains__('Quark') and k.__contains__('Central'):
            Central_Quark_unumpy += v
        elif k.__contains__('Gluon') and k.__contains__('Central'):
            Central_Gluon_unumpy += v
    return Forward_unumpy, Central_unumpy, Quark_unumpy, Gluon_unumpy, Forward_Quark_unumpy, Forward_Gluon_unumpy, Central_Quark_unumpy, Central_Gluon_unumpy
    

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

def Calcu_Frac_unumpy(Forward_Quark, Central_Quark, Forward, Central):
    try:
        frac_Forward_Quark = np.sum(Forward_Quark) / np.sum(Forward)
        frac_Central_Quark = np.sum(Central_Quark) / np.sum(Central)
    except RuntimeWarning:
        print(Forward)
        print(Central)

    frac_Forward_Gluon = 1 - frac_Forward_Quark
    frac_Central_Gluon = 1 - frac_Central_Quark

    f = np.array([[frac_Forward_Quark,  frac_Forward_Gluon], [frac_Central_Quark, frac_Central_Gluon]])

    return f, unumpy.ulinalg.inv(f)

def Normalize_unumpy(array_unumpy, bin_width=1.0):
    area = np.sum(unumpy.nominal_values(array_unumpy)) * bin_width
    return array_unumpy / area

def safe_array_divide_unumpy(numerator, denominator):
    if 0 in unumpy.nominal_values(denominator):
        raise Exception(f"0 exists in the denominator for unumpy, check it!")
    else:
        ratio = np.true_divide(numerator, denominator)        
    return ratio

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

def Plot_Extracted_unumpy(pt, var, output_path, period, reweighting_var, reweighting_factor, p_Quark, extract_p_Quark, p_Gluon, extract_p_Gluon, extract_p_Quark_Data, extract_p_Gluon_Data,
                          show_yields=False, n_Forward_MC=None, n_Central_MC=None, n_Forward_Data=None, n_Central_Data=None):
    bin_edges = GetHistBin(var)
    bin_centers = 1/2 * (bin_edges[:-1] + bin_edges[1:])
        
    jet_types = ["quark", "gluon"]
    color_types = ["blue", "red"]
    plot_data = [[p_Quark, extract_p_Quark, extract_p_Quark_Data], 
                 [p_Gluon, extract_p_Gluon, extract_p_Gluon_Data]]
    plot_data_bin_content = unumpy.nominal_values(plot_data) 
    plot_data_bin_error = unumpy.std_devs(plot_data)

    for i, jet_type in enumerate(jet_types):  # i is the idx of jet type
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
        # ax0.stairs(values = plot_data[i][0], edges = bin_edges, color = color_types[i], label = f'{jet_type}, extracted MC', baseline=None)
        # ax0.stairs(values = plot_data[i][1], edges = bin_edges, color = color_types[i], linestyle='--', label = f'{jet_type}, truth MC', baseline=None)
        # ax0.stairs(values = plot_data[i][2], edges = bin_edges, color = color_types[i], linestyle=':', label = f'{jet_type}, extracted Data', baseline=None)
        ax0.errorbar(x = bin_centers, y = plot_data_bin_content[i][0], yerr = plot_data_bin_error[i][0], drawstyle = 'steps-mid', label = "Truth MC")
        ax0.errorbar(x = bin_centers, y = plot_data_bin_content[i][1], yerr = plot_data_bin_error[i][1], drawstyle = 'steps-mid', label = "Extracted MC")
        ax0.errorbar(x = bin_centers, y = plot_data_bin_content[i][2], yerr = plot_data_bin_error[i][2], drawstyle = 'steps-mid', label = "Extracted Data", color= "black", linestyle='', marker= "o")
        ax0.legend()

        y_max = np.max(plot_data_bin_content)
        ax0.set_ylim(-0.01, y_max * 1.2)
        ax0.set_ylabel("Normalized")
        ax0.set_title(f"{pt} GeV {jet_type}: Extracted " + rf"{Map_var_title[var]}"  + f" distro, {reweighting_factor}")
        if show_yields:
            ax0.text(x=0.0, y=0.04, 
            s = f"MC forward yield:{n_Forward_MC:.2e},central yield:{n_Central_MC:.2e} \n"+
                f"Data forward yield:{n_Forward_Data:.2e}, central yield:{n_Central_Data:.2e}",
            ha='left', va='bottom', transform=ax0.transAxes)
        
        ratio_truthMC_over_extractedMC = safe_array_divide_unumpy(plot_data[i][0], plot_data[i][1])
        ratio_data_over_extractedMC = safe_array_divide_unumpy(plot_data[i][2], plot_data[i][1])
        ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(ratio_truthMC_over_extractedMC), yerr = unumpy.std_devs(ratio_truthMC_over_extractedMC), drawstyle = 'steps-mid', label = "Truth MC / Extracted MC")
        ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(ratio_data_over_extractedMC), yerr = unumpy.std_devs(ratio_data_over_extractedMC), drawstyle = 'steps-mid', label = "Extracted Data / Extracted MC", color= "black", linestyle='', marker= "o")

        # plot_data[i][1][plot_data[i][1]==0] = np.inf
        # ax1.stairs(values = plot_data[i][0]/plot_data[i][1] , edges=bin_edges, color = color_types[i], linestyle='--', label = 'Extracted MC / Truth MC', baseline=None)
        # ax1.stairs(values = plot_data[i][2]/plot_data[i][1] , edges=bin_edges, color = color_types[i], linestyle=':', label = 'Extracted Data / Truth MC', baseline=None)
        ax1.legend()
        ax1.set_ylim(0.7,1.3)
        ax1.set_ylabel("Ratio")
        ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')
        output_path_new = output_path / period / f"{reweighting_var}_{reweighting_factor}"  / var 
        if not output_path_new.exists():
            output_path_new.mkdir(parents = True)

        fig.tight_layout()
        fig.savefig( output_path_new / f"Eff_Rejection_extracted_{pt}_{jet_type}_{var}.jpg", dpi=100)
        plt.close()

def Plot_WP(WP, var, output_path, period, reweighting_var, reweighting_factor,
            quark_effs, gluon_rejs, quark_effs_data, gluon_rejs_data):
    bin_edges = np.array([500, 600, 800, 1000, 1200, 1500, 2000])
    bin_centers = 1/2 * (bin_edges[:-1] + bin_edges[1:])

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(quark_effs), yerr = unumpy.std_devs(quark_effs), drawstyle = 'steps-mid', label = "Quark Efficiency, Extracted MC", color = "blue")
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(gluon_rejs), yerr = unumpy.std_devs(gluon_rejs), drawstyle = 'steps-mid', label = "Gluon Rejection, Extracted MC", color = "red")
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(quark_effs_data), yerr = unumpy.std_devs(quark_effs_data), drawstyle = 'steps-mid', label = "Quark Efficiency, Extracted Data", color= "blue", linestyle='--', marker= "o")
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(gluon_rejs_data), yerr = unumpy.std_devs(gluon_rejs_data), drawstyle = 'steps-mid', label = "Gluon Rejection, Extracted Data",color= "red", linestyle='--', marker= "o")
    ax0.legend()
    ax0.set_yticks(np.linspace(0, 1, 21))
    ax0.set_xticks(bin_edges)
    ax0.set_ylim(0.3, 1.2)
    # ax0.set_xlim(bin_centers[0], bin_centers[-1])

    ax0.grid()
    ax0.set_title(f"{reweighting_var} for extracted q/g at {WP} WP")

    SF_quark = safe_array_divide_unumpy(quark_effs_data, quark_effs)
    SF_gluon = safe_array_divide_unumpy(gluon_rejs_data, gluon_rejs)
    ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(SF_quark), yerr = unumpy.std_devs(SF_quark), drawstyle = 'steps-mid', label = "SF for quark")
    ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(SF_gluon), yerr = unumpy.std_devs(SF_gluon), drawstyle = 'steps-mid', label = "SF for gluon")
    ax1.set_ylim(0.7, 1.3)
    # ax1.set_xlim(bin_centers[0], bin_centers[-1])
    print(f"{WP}:")
    print(f"Quark effs for Extracted MC:\n")
    print(quark_effs)

    print(f"Quark effs for Extracted Data:\n")
    print(quark_effs_data)


    output_path_new = output_path / period / "WPs" / f"{reweighting_var}_{reweighting_factor}" /var
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True)
    fig.savefig( output_path_new/ f"{reweighting_var}_WP_{WP}.jpg", dpi=300)
    plt.close()
    

def Read_Histogram_Root(file, sampletype="MC", code_version="new", reweighting_var=None, reweighting_factor="none"):
    """A general func to read the contents of a root file. In future we'll discard the root format.

    Args:
        file (str): the path to the file you want to read
        sampletype (str, optional): MC or Data. Jet type not known in Data. Defaults to "MC".
        code_version (str, optional): new or old. new is being developed. Defaults to "new".
        reweighting_var (str, optional): ntrk or bdt. Defaults to None.
        reweighting_factor (str, optional): quark or gluon. Defaults to "none".

    Returns:
        (Dict, Dict): Return HistMap and HistMap_Error. 
    """
    # defile which TDirectory to look at based on {reweighting_var}_{reweighting_factor}
    reweighting_map = {
        "none" : "NoReweighting",
        "quark" : "Reweighting_Quark_Factor",
        "gluon" : "Reweighting_Gluon_Factor"
    }

    if sampletype== "MC":
        label_jettype = ["Quark", "Gluon", "B_Quark", "C_Quark", "Other"]
    elif sampletype == "Data":
        label_jettype = ["Data"]
    
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    label_leadingtype = ["LeadingJet", "SubLeadingJet"]
    label_etaregion = ["Forward", "Central"]
    label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]

    HistMap = {}
    HistMap_Error = {}
    HistMap_unumpy = {}

    if code_version=="new":
        if reweighting_var == None:
            TDirectory_name = reweighting_map[reweighting_factor]
        else:
            TDirectory_name = f"{reweighting_var}_" + reweighting_map[reweighting_factor]

        file = uproot.open(file)[TDirectory_name]
    
    avail_keys = [*file.keys()]
    for pt in label_ptrange[:-1]:
        for leadingtype in label_leadingtype:
            for eta_region in label_etaregion: 
                for var in label_var:
                    for jettype in label_jettype:

                        key = f"{pt}_{leadingtype}_{eta_region}_{jettype}_{var}"
                        if (key in avail_keys) or (key+";1" in avail_keys):
                            HistMap[key] = file[key].to_numpy()[0]
                            HistMap_Error[key] = file[f"{key}_err"].to_numpy()[0] # No more suffix '_err' in HistMap_Error

    
    for key, value in HistMap.items():
        HistMap_unumpy[key] = unumpy.uarray(value, np.sqrt(HistMap_Error[key]))
    return HistMap, HistMap_Error, HistMap_unumpy

def Eff_Rejection(input_mc_path, input_data_path, period, reweighting_factor, output_path):
    # define some variables to make up the histogram names 
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    # label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
    # label_var = ['bdt', 'ntrk']
    label_var = ['ntrk']
    label_leadingtype = ["LeadingJet", "SubLeadingJet"]
    label_etaregion = ["Forward", "Central"]
    label_type = ["Gluon", "Quark", "B_Quark", "C_Quark"]

    # defile which TDirectory to look at based on {reweighting_factor}
    reweighting_map = {
        "none" : "NoReweighting",
        "quark" : "Reweighting_Quark_Factor",
        "gluon" : "Reweighting_Gluon_Factor"
    }

    WPs = [0.5, 0.6, 0.7, 0.8]
    for WP in WPs:
        for reweighting_var in ["ntrk"]:
            HistMap_MC, HistMap_Error_MC, HistMap_MC_unumpy = Read_Histogram_Root(input_mc_path, sampletype="MC", code_version="new", reweighting_var=reweighting_var, reweighting_factor="quark")
            HistMap_Data, HistMap_Error_Data, HistMap_Data_unumpy = Read_Histogram_Root(input_data_path, sampletype="Data", code_version="new", reweighting_var=reweighting_var, reweighting_factor="quark")

            for var in label_var:
                quark_effs_at_pt = []
                gluon_rejs_at_pt = []
                quark_effs_data_at_pt = []
                gluon_rejs_data_at_pt = []
                for ii, l_pt in enumerate(label_ptrange[:-1]):

                    sel_HistMap_MC_Manual = {}
                    sel_HistMap_Error_MC_Manual = {}

                    sel_HistMap_Data_Manual = {}
                    sel_HistMap_Error_Data_Manual = {}

                    sel_HistMap_MC_unumpy = {}
                    sel_HistMap_Data_unumpy = {}

                    for i, l_leadingtype  in enumerate(label_leadingtype):
                        for j, l_etaregion in enumerate(label_etaregion):
                            key_data = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + "Data" + "_" + var
                            sel_HistMap_Data_Manual[key_data] = HistMap_Data[key_data]
                            sel_HistMap_Error_Data_Manual[key_data] = HistMap_Error_Data[key_data]
                            sel_HistMap_Data_unumpy[key_data] = HistMap_Data_unumpy[key_data]

                            for k, l_type in enumerate(label_type):
                                key_mc = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + l_type + "_" + var
                                sel_HistMap_MC_Manual[key_mc] = HistMap_MC[key_mc]
                                sel_HistMap_Error_MC_Manual[key_mc] = HistMap_Error_MC[key_mc]
                                sel_HistMap_MC_unumpy[key_mc] = HistMap_MC_unumpy[key_mc]
                    # The following two lines left for check the mannual calclulation 
                    # Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct(HistMap=sel_HistMap_MC_Manual, HistMap_Error=sel_HistMap_Error_MC_Manual, n_bins = len(GetHistBin(var)) - 1, sampletype="MC")
                    # Forward, Central = Construct(HistMap=sel_HistMap_Data_Manual, HistMap_Error=sel_HistMap_Error_Data_Manual, n_bins = len(GetHistBin(var)) - 1, sampletype="Data")

                    Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct_unumpy(HistMap_unumpy=sel_HistMap_MC_unumpy, n_bins = len(GetHistBin(var)) - 1, sampletype="MC")
                    Forward_Data, Central_Data = Construct_unumpy(HistMap_unumpy=sel_HistMap_Data_unumpy, n_bins = len(GetHistBin(var)) - 1, sampletype="Data")

                    f, f_inv = Calcu_Frac_unumpy(Forward_Quark, Central_Quark, Forward, Central)
                    # normalize 
                    ## Truth
                    p_Quark = Normalize_unumpy(Quark)
                    p_Gluon = Normalize_unumpy(Gluon)

                    p_Forward = Normalize_unumpy(Forward)
                    p_Central = Normalize_unumpy(Central)
                    p_Forward_Data = Normalize_unumpy(Forward_Data)
                    p_Central_Data = Normalize_unumpy(Central_Data)

                    # for stats purpose 
                    n_Forward_MC = np.sum(unumpy.nominal_values(Forward))
                    n_Central_MC = np.sum(unumpy.nominal_values(Central))
                    n_Forward_Data = np.sum(unumpy.nominal_values(Forward_Data))
                    n_Central_Data = np.sum(unumpy.nominal_values(Central_Data))
                    

                    extract_p_Quark = f_inv[0][0] * p_Forward + f_inv[0][1]* p_Central 
                    extract_p_Gluon = f_inv[1][0] * p_Forward + f_inv[1][1]* p_Central 

                    extract_p_Quark_Data = f_inv[0][0] * p_Forward_Data + f_inv[0][1]* p_Central_Data 
                    extract_p_Gluon_Data = f_inv[1][0] * p_Forward_Data + f_inv[1][1]* p_Central_Data 

                    extract_p_Quark_cum_sum = np.cumsum(unumpy.nominal_values(extract_p_Quark))
                    cut = np.where(extract_p_Quark_cum_sum >= WP)[0][0]+1
                    

                    quark_effs_at_pt.append(np.sum(extract_p_Quark[:cut])) 
                    gluon_rejs_at_pt.append(np.sum(extract_p_Gluon[cut:]))
                    quark_effs_data_at_pt.append(np.sum(extract_p_Quark_Data[:cut]))
                    gluon_rejs_data_at_pt.append(np.sum(extract_p_Gluon_Data[cut:]))
                
                Plot_WP(WP = WP, var= var, output_path= output_path, 
                        period= period, reweighting_var = reweighting_var,
                        reweighting_factor= reweighting_map[reweighting_factor],
                        quark_effs= quark_effs_at_pt, gluon_rejs = gluon_rejs_at_pt,
                        quark_effs_data=quark_effs_data_at_pt, gluon_rejs_data = gluon_rejs_data_at_pt)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script does the MC Closure test. ')
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

    Eff_Rejection(input_mc_path=mc_file_path, input_data_path=data_file_path, period = period, reweighting_factor = args.reweighting , output_path = output_path)
    

