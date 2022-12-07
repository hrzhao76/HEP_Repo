import argparse
from genericpath import exists
import uproot
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from pathlib import Path
import pickle


from uncertainties import ufloat, unumpy
import atlas_mpl_style as ampl
ampl.use_atlas_style(usetex=False)

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
        _denominator_nominal_values = unumpy.nominal_values(denominator)
        _denominator_std_devs = unumpy.std_devs(denominator)
        zero_idx = np.where(_denominator_nominal_values==0)[0]
        _denominator_nominal_values[zero_idx] = np.inf
        _denominator_std_devs[zero_idx] = 0 
        _denominator = unumpy.uarray(_denominator_nominal_values, _denominator_std_devs)

        ratio = np.true_divide(numerator, _denominator) 
        # raise Warning(f"0 exists in the denominator for unumpy, check it!")
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
        if reweighting_factor == "none":
            TDirectory_name = reweighting_map[reweighting_factor]
        else:
            TDirectory_name = f"{reweighting_var}_" + reweighting_map[reweighting_factor]

        file = uproot.open(file)[TDirectory_name]
    elif code_version=="old":
        file = uproot.open(file)
    
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

def Extract(HistMap_MC_unumpy, HistMap_Data_unumpy):
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    # label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt"]
    label_var = ['ntrk', 'bdt']
    # label_var = ['ntrk']
    label_leadingtype = ["LeadingJet", "SubLeadingJet"]
    label_etaregion = ["Forward", "Central"]
    label_type = ["Gluon", "Quark", "B_Quark", "C_Quark"]

    # HistMap_MC, HistMap_Error_MC, HistMap_MC_unumpy = Read_Histogram_Root(input_mc_path, sampletype="MC", code_version="new", reweighting_var=reweighting_var, reweighting_factor="quark")
    # HistMap_Data, HistMap_Error_Data, HistMap_Data_unumpy = Read_Histogram_Root(input_data_path, sampletype="Data", code_version="new", reweighting_var=reweighting_var, reweighting_factor="quark")
    Extraction_Results = {}
    for var in label_var:
        Extraction_Results[var] = {}
        for l_pt in label_ptrange[:-1]:

            sel_HistMap_MC_unumpy = {}
            sel_HistMap_Data_unumpy = {}

            for i, l_leadingtype  in enumerate(label_leadingtype):
                for j, l_etaregion in enumerate(label_etaregion):
                    key_data = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + "Data" + "_" + var
                    sel_HistMap_Data_unumpy[key_data] = HistMap_Data_unumpy[key_data]

                    for k, l_type in enumerate(label_type):
                        key_mc = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + l_type + "_" + var
                        sel_HistMap_MC_unumpy[key_mc] = HistMap_MC_unumpy[key_mc]
            # The following two lines left for check the mannual calclulation 
            # Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct(HistMap=sel_HistMap_MC_Manual, HistMap_Error=sel_HistMap_Error_MC_Manual, n_bins = len(GetHistBin(var)) - 1, sampletype="MC")
            # Forward, Central = Construct(HistMap=sel_HistMap_Data_Manual, HistMap_Error=sel_HistMap_Error_Data_Manual, n_bins = len(GetHistBin(var)) - 1, sampletype="Data")

            Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct_unumpy(HistMap_unumpy=sel_HistMap_MC_unumpy, n_bins = len(GetHistBin(var)) - 1, sampletype="MC")
            Forward_Data, Central_Data = Construct_unumpy(HistMap_unumpy=sel_HistMap_Data_unumpy, n_bins = len(GetHistBin(var)) - 1, sampletype="Data")

            # f, f_inv = Calcu_Frac_unumpy(Forward_Quark, Central_Quark, Forward, Central)
            f, f_inv = Calcu_Frac(unumpy.nominal_values(Forward_Quark), unumpy.nominal_values(Central_Quark), unumpy.nominal_values(Forward), unumpy.nominal_values(Central))
            # normalize 
            ## Truth
            p_Quark = Normalize_unumpy(Quark)
            p_Gluon = Normalize_unumpy(Gluon)

            p_Forward = Normalize_unumpy(Forward)
            p_Central = Normalize_unumpy(Central)
            p_Forward_Data = Normalize_unumpy(Forward_Data)
            p_Central_Data = Normalize_unumpy(Central_Data)
            
            extract_p_Quark = f_inv[0][0] * p_Forward + f_inv[0][1]* p_Central 
            extract_p_Gluon = f_inv[1][0] * p_Forward + f_inv[1][1]* p_Central 

            extract_p_Quark_Data = f_inv[0][0] * p_Forward_Data + f_inv[0][1]* p_Central_Data 
            extract_p_Gluon_Data = f_inv[1][0] * p_Forward_Data + f_inv[1][1]* p_Central_Data 
        
            Extraction_Results[var][l_pt] = {
                "Forward_MC": Forward,
                "Central_MC": Central,
                "Forward_Data": Forward_Data,
                "Central_Data": Central_Data,
                "p_Quark": p_Quark,
                "p_Gluon": p_Gluon,
                "extract_p_Quark_MC": extract_p_Quark,
                "extract_p_Gluon_MC": extract_p_Gluon,
                "extract_p_Quark_Data": extract_p_Quark_Data,
                "extract_p_Gluon_Data": extract_p_Gluon_Data
            }

    return Extraction_Results


def Plot_ForwardCentral_MCvsData(pt, var, output_path, period, reweighting_var, reweighting_option, 
                                 Forward_MC, Central_MC, Forward_Data, Central_Data, if_norm, show_yields=True):

    bin_edges = GetHistBin(var)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Forward_MC), yerr=unumpy.std_devs(Forward_MC), color = 'blue', label = 'Forward MC', drawstyle='steps-mid')
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Central_MC), yerr=unumpy.std_devs(Central_MC), color = 'red', label = 'Central MC', drawstyle='steps-mid')
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Forward_Data), yerr=unumpy.std_devs(Forward_Data), color = 'blue', label = 'Forward Data', marker='.', linestyle="none")
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Central_Data), yerr=unumpy.std_devs(Central_Data), color = 'red', label = 'Central Data', marker='.', linestyle="none")    
    ax0.legend()
    ax0.set_title(f"{pt} GeV: MC vs Data " + rf"{Map_var_title[var]}"  + f" distro, {reweighting_option}")
    if show_yields and not if_norm:
        n_Forward_MC = np.sum(unumpy.nominal_values(Forward_MC))
        n_Central_MC = np.sum(unumpy.nominal_values(Central_MC))
        n_Forward_Data = np.sum(unumpy.nominal_values(Forward_Data))
        n_Central_Data = np.sum(unumpy.nominal_values(Central_Data))
        ax0.text(x=0.3, y=0.04, 
                s = f"MC forward yield:{n_Forward_MC:.2e},central yield:{n_Central_MC:.2e} \n"+
                    f"Data forward yield:{n_Forward_Data:.2e}, central yield:{n_Central_Data:.2e}",
                    ha='left', va='bottom', transform=ax0.transAxes)
        

    ratio_Forward = safe_array_divide_unumpy(numerator = Forward_Data, denominator = Forward_MC)
    ratio_Central = safe_array_divide_unumpy(numerator = Central_Data, denominator = Central_MC)

    ax1.errorbar(x=bin_centers, y=unumpy.nominal_values(ratio_Forward), yerr=unumpy.std_devs(ratio_Forward), color = 'blue', label = 'Forward', drawstyle='steps-mid')
    ax1.errorbar(x=bin_centers, y=unumpy.nominal_values(ratio_Central), yerr=unumpy.std_devs(ratio_Central), color = 'red', label = 'Central', drawstyle='steps-mid')
    ax1.set_ylabel("Data/MC")
    ax1.set_ylim(0.7, 1.3)
    ax1.legend()
    ax1.set_xlabel(f"{Map_var_title[var]}")
    ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--', label='ratio = 1')
    ax1.plot()

    output_path_new = output_path / period / "FvsC" / f"{reweighting_var}_{reweighting_option}" / var 
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True, exist_ok =True)
    if if_norm == True:
        ax0.set_ylabel("Normalized")
        fig_name = output_path_new / f"MCvsData_FvsC_{pt}_{var}_{reweighting_option}_normed.jpg"
    else: 
        ax0.set_ylabel("Yields")
        fig_name = output_path_new / f"MCvsData_FvsC_{pt}_{var}_{reweighting_option}.jpg"
    fig.savefig(fig_name)
    plt.close()

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
        ax0.set_ylim(-0.01, y_max * 1.3)
        ax0.set_ylabel("Normalized")
        ax0.set_title(f"{pt} GeV {jet_type}: Extracted " + rf"{Map_var_title[var]}"  + f" distro, {reweighting_factor}")
        ampl.draw_atlas_label(0.1, 0.85, ax=ax0)
        if show_yields:
            ax0.text(x=0.3, y=0.04, 
            s = f"MC forward yield:{n_Forward_MC:.2e},central yield:{n_Central_MC:.2e} \n"+
                f"Data forward yield:{n_Forward_Data:.2e}, central yield:{n_Central_Data:.2e}",
            ha='left', va='bottom', transform=ax0.transAxes)
        # breakpoint()
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
        output_path_new = output_path / period / "Extractions" /f"{reweighting_var}_{reweighting_factor}"  / var 
        if not output_path_new.exists():
            output_path_new.mkdir(parents = True)

        fig.tight_layout()
        fig.savefig( output_path_new / f"DataExtraction_{pt}_{jet_type}_{var}.jpg")
        plt.close()

def Plot_WP(WP, var, output_path, period, reweighting_var, reweighting_factor,
            quark_effs, gluon_rejs, quark_effs_data, gluon_rejs_data):
    bin_edges = np.array([500, 600, 800, 1000, 1200, 1500, 2000])
    bin_centers = 1/2 * (bin_edges[:-1] + bin_edges[1:])

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(quark_effs), yerr = unumpy.std_devs(quark_effs), label = "Quark Efficiency, Extracted MC", color = "blue",linestyle='none', marker='^')
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(gluon_rejs), yerr = unumpy.std_devs(gluon_rejs), label = "Gluon Rejection, Extracted MC", color = "red",linestyle='none', marker='^')
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(quark_effs_data), yerr = unumpy.std_devs(quark_effs_data), label = "Quark Efficiency, Extracted Data", color= "blue", linestyle='none', marker= "o")
    ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(gluon_rejs_data), yerr = unumpy.std_devs(gluon_rejs_data), label = "Gluon Rejection, Extracted Data",color= "red", linestyle='none', marker= "o")
    ax0.legend()
    ax0.set_yticks(np.linspace(0, 1, 21))
    ax0.set_xticks(bin_edges)
    ax0.set_ylim(0.4, 1.2)
    ax0.set_xlim(bin_edges[0], bin_edges[-1])
    ax0.set_ylabel("Efficiency or Rejection")

    ax0.grid()
    ax0.set_title(f"{var} for extracted q/g at {WP} WP")
    ampl.draw_atlas_label(0.1, 0.9, ax=ax0)

    SF_quark = safe_array_divide_unumpy(quark_effs_data, quark_effs)
    SF_gluon = safe_array_divide_unumpy(gluon_rejs_data, gluon_rejs)
    ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(SF_quark), yerr = unumpy.std_devs(SF_quark), linestyle='none', label = "quark SF", marker='.')
    ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(SF_gluon), yerr = unumpy.std_devs(SF_gluon), linestyle='none', label = "gluon SF", marker='.')
    ax1.legend(fontsize = 'x-small')
    ax1.set_ylim(0.7, 1.3)
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    ax1.set_xticks(bin_edges)
    ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'gray', linestyle = '--')
    ax1.set_ylabel("SFs")

    # print(f"{WP}:")
    # print(f"Quark effs for Extracted MC:\n")
    # print(quark_effs)

    # print(f"Quark effs for Extracted Data:\n")
    # print(quark_effs_data)
    # print(f"SF for quark:\n")

    # print(SF_quark)


    output_path_new = output_path / period / "WPs" / f"{reweighting_var}_{reweighting_factor}" / var
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True)
    fig.savefig( output_path_new/ f"{reweighting_var}_WP_{WP}.jpg")
    plt.close()

    return SF_quark, SF_gluon

def WriteSFtoPickle(var, Hist_SFs, output_path, period, reweighting_var, reweighting_factor ):
    output_path_new = output_path / period / "SFs_pkls" / f"{reweighting_var}_{reweighting_factor}" / var

    if not output_path_new.exists():
        output_path_new.mkdir(parents = True)
    
    pkl_file_name = output_path_new / f"SFs.pkl"
    print(f"Writing Scale Factors to the pickle file: {pkl_file_name}")
    with open(pkl_file_name, "wb") as out_pkl:
        pickle.dump(Hist_SFs, out_pkl)



def Calculate_SF(input_mc_path, input_data_path, period, reweighting_factor, output_path):
    label_var = ['ntrk', 'bdt']
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    reweighting_map = {
        "none" : "NoReweighting",
        "quark" : "Reweighting_Quark_Factor",
        "gluon" : "Reweighting_Gluon_Factor"
    }

    for reweighting_var in ["ntrk", "bdt"]:
        HistMap_MC, HistMap_Error_MC, HistMap_MC_unumpy = Read_Histogram_Root(input_mc_path, sampletype="MC", code_version="new", reweighting_var=reweighting_var, reweighting_factor=reweighting_factor)
        HistMap_Data, HistMap_Error_Data, HistMap_Data_unumpy = Read_Histogram_Root(input_data_path, sampletype="Data", code_version="new", reweighting_var=reweighting_var, reweighting_factor=reweighting_factor)
        Extraction_Results = Extract(HistMap_MC_unumpy, HistMap_Data_unumpy)
        WPs = [0.5, 0.6, 0.7, 0.8]
        SFs = {}

        for var in label_var:
            SFs[var] = {}
            for l_pt in label_ptrange[:-1]:
                Extraction_var_pt =  Extraction_Results[var][l_pt]
                #### Draw Forward vs Central plots 
                Plot_ForwardCentral_MCvsData(pt = l_pt, var= var, output_path= output_path, 
                                    period= period, reweighting_var = reweighting_var,
                                    reweighting_option= reweighting_map[reweighting_factor],
                                    Forward_MC= Extraction_var_pt['Forward_MC'], 
                                    Central_MC= Extraction_var_pt['Central_MC'],
                                    Forward_Data= Extraction_var_pt['Forward_Data'], 
                                    Central_Data= Extraction_var_pt['Central_Data'],
                                    if_norm=False, show_yields=True)

                Plot_ForwardCentral_MCvsData(pt = l_pt, var= var, output_path= output_path, 
                                    period= period, reweighting_var = reweighting_var,
                                    reweighting_option= reweighting_map[reweighting_factor],
                                    Forward_MC= Normalize_unumpy(Extraction_var_pt['Forward_MC']), 
                                    Central_MC= Normalize_unumpy(Extraction_var_pt['Central_MC']),
                                    Forward_Data= Normalize_unumpy(Extraction_var_pt['Forward_Data']), 
                                    Central_Data= Normalize_unumpy(Extraction_var_pt['Central_Data']),
                                    if_norm=True, show_yields=False)

                #### Draw extraction plots 
                Plot_Extracted_unumpy(pt = l_pt, var= var, output_path= output_path, 
                                        period= period, reweighting_var = reweighting_var,
                                        reweighting_factor= reweighting_map[reweighting_factor],
                                        p_Quark=Extraction_var_pt['p_Quark'], p_Gluon=Extraction_var_pt['p_Gluon'],
                                        extract_p_Quark = Extraction_var_pt['extract_p_Quark_MC'], extract_p_Gluon = Extraction_var_pt['extract_p_Gluon_MC'],
                                        extract_p_Quark_Data = Extraction_var_pt['extract_p_Quark_Data'], extract_p_Gluon_Data = Extraction_var_pt['extract_p_Gluon_Data'],
                                        show_yields=True, 
                                        n_Forward_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_MC'])), 
                                        n_Central_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_MC'])), 
                                        n_Forward_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_Data'])), 
                                        n_Central_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_Data'])))
            #### Draw working points 
            for WP in WPs:
                SFs[var][WP] = {}
                quark_effs_at_pt = []
                gluon_rejs_at_pt = []
                quark_effs_data_at_pt = []
                gluon_rejs_data_at_pt = []
                for ii, l_pt in enumerate(label_ptrange[:-1]):
                    extract_p_Quark_MC =  Extraction_Results[var][l_pt]['extract_p_Quark_MC']
                    extract_p_Gluon_MC =  Extraction_Results[var][l_pt]['extract_p_Gluon_MC']
                    extract_p_Quark_Data =  Extraction_Results[var][l_pt]['extract_p_Quark_Data']
                    extract_p_Gluon_Data =  Extraction_Results[var][l_pt]['extract_p_Gluon_Data']

                    extract_p_Quark_cum_sum = np.cumsum(unumpy.nominal_values(extract_p_Quark_MC))
                    cut = np.where(extract_p_Quark_cum_sum >= WP)[0][0]+1
                    

                    quark_effs_at_pt.append(np.sum(extract_p_Quark_MC[:cut])) 
                    gluon_rejs_at_pt.append(np.sum(extract_p_Gluon_MC[cut:]))
                    quark_effs_data_at_pt.append(np.sum(extract_p_Quark_Data[:cut]))
                    gluon_rejs_data_at_pt.append(np.sum(extract_p_Gluon_Data[cut:]))

                SF_quark, SF_gluon = Plot_WP(WP = WP, var= var, output_path= output_path, 
                        period= period, reweighting_var = reweighting_var,
                        reweighting_factor= reweighting_map[reweighting_factor],
                        quark_effs= quark_effs_at_pt, gluon_rejs = gluon_rejs_at_pt,
                        quark_effs_data=quark_effs_data_at_pt, gluon_rejs_data = gluon_rejs_data_at_pt)
                SFs[var][WP]["Quark"] = SF_quark
                SFs[var][WP]["Gluon"] = SF_gluon

            WriteSFtoPickle(var = var,Hist_SFs = SFs, output_path=output_path, period=period, reweighting_var = reweighting_var,
                        reweighting_factor= reweighting_map[reweighting_factor])


if __name__ == '__main__':
    """This script do the matrix method and calculate SFs and save them to pickle files. 
       It generates the following structure for a successful run. 
       <output-path>
           └── <period>
                ├── Extractions
                ├── FvsC
                ├── SFs_pkls
                └── WPs

    Raises:
        Exception: if the mc_file is not a root file, raise an error. 
        Exception: if the input file is not consistent with the period passed, raise an error. 
    """
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

    Calculate_SF(input_mc_path=mc_file_path, input_data_path=data_file_path, period = period, reweighting_factor = args.reweighting , output_path = output_path)
    
