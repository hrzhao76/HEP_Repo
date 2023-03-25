import joblib 
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import hist
from hist import Hist 
from uncertainties import ufloat, unumpy
from tqdm import tqdm
import re
import argparse

import sys
core_code_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow'
sys.path.append(core_code_path)

from core.Calculate_SF import convert_histdict2unumpy, Construct_unumpy, Normalize_unumpy, Plot_WP, WriteSFtoPickle
from core.utils import HistBins, label_var, label_pt_bin

nominal_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/nominal'
output_path = ''
def construct_Forward_Central(data_pred_hist_period_event_weight):
    label_var = ['jet_pt', 'jet_eta', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'jet_trackBDT', 'GBDT_newScore']

    label_leadingtype = ["LeadingJet", "SubLeadingJet"]
    label_etaregion = ["Forward", "Central"]
    Data_period = dict.fromkeys(label_var)
    
    for var in label_var:
        Data_period[var] = dict.fromkeys(label_pt_bin[:-1])
        for l_pt in label_pt_bin[:-1]:
            sel_HistMap_Data_unumpy = {}
            for i, l_leadingtype  in enumerate(label_leadingtype):
                for j, l_etaregion in enumerate(label_etaregion):
                    key_data = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + "Data" + "_" + var
                    sel_HistMap_Data_unumpy[key_data] = data_pred_hist_period_event_weight[key_data]

                    Forward_Data, Central_Data = Construct_unumpy(HistMap_unumpy=sel_HistMap_Data_unumpy, n_bins = len(HistBins[var]) - 1, sampletype="Data")
            
            Data_period[var][l_pt] = {
                "Forward_Data": Forward_Data,
                "Central_Data": Central_Data,
            } 
    
    return Data_period

def bootstrap_data(nominal_Data, period):
    bootstrap_Data = dict.fromkeys(nominal_Data.keys())
    
    for var, data_var in nominal_Data.items():
        bootstrap_Data[var] = dict.fromkeys(data_var.keys())
        for pt, data_pt in data_var.items():
            bootstrap_Data[var][pt] = dict.fromkeys(data_pt.keys())
            for region, data_region in data_pt.items():
                hist_var_pt_region_unumpy = unumpy.nominal_values(data_region)
                bootstrap = np.random.poisson(lam=hist_var_pt_region_unumpy)

                if period == "18":
                    variances = 58.45/39.91 ** 2 * bootstrap
                else: 
                    variances = bootstrap

                bootstrap_Data[var][pt][region] =  unumpy.uarray(bootstrap, np.sqrt(variances))
    return bootstrap_Data

def merged_bootstrap_period(bootstrap_data):
    keys = [*bootstrap_data.keys()]
    
    bootstrap_data_merged = bootstrap_data[keys[0]].copy()

    for bootstrap_data_key in keys[1:]:
        bootstrap_data_period = bootstrap_data[bootstrap_data_key]
        for var, data_var in bootstrap_data_period.items():
            for pt, data_pt in data_var.items():
                for region, data_region in data_pt.items():
                    bootstrap_data_merged[var][pt][region] += data_region
    
    return bootstrap_data_merged

def generate_nominal(nominal_path:Path):
    data_Forward_Central_periods = dict.fromkeys(['1516', '17', '18'])
    for file in sorted(nominal_path.rglob("data*_pred_hist*")):
        period_search_pattern = r"data(\d+)_pred_hists"
        period = re.search(period_search_pattern, file.stem).group((1))
        assert period in ["1516", "17", "18"]
        
        data_pred_hist_period = joblib.load(file)
        data_pred_hist_period_event_weight = convert_histdict2unumpy(data_pred_hist_period['event_weight']) 
        data_Forward_Central_periods[period] = construct_Forward_Central(data_pred_hist_period_event_weight)

    return data_Forward_Central_periods

def calculate_sf():
    reweight_factor_path = nominal_path / "reweight_factor.pkl"
    reweight_factor = joblib.load(reweight_factor_path)

    Extraction_Results_bootstrap = {}

    SF_label_vars = ['jet_nTracks', 'jet_trackBDT', 'GBDT_newScore']
    WPs = [0.5, 0.6, 0.7, 0.8]
    for reweight_var in SF_label_vars:

        Extraction_Results_bootstrap[reweight_var] = {}
        # Load information from MC 
        Extraction_Results_path = nominal_path / "plots" / "ADE" / "Extraction_Results" / f"{reweight_var}_Extraction_Results.pkl"
        Extraction_Results = joblib.load(Extraction_Results_path)

        for l_pt in label_pt_bin[:-1]:
            reweight_factor_pt_var = reweight_factor[l_pt][reweight_var]['quark_factor']

            f_inv = Extraction_Results[reweight_var][l_pt]['f_inv']
            
            Forward_Data = bootstrap_data[reweight_var][l_pt]['Forward_Data']
            Central_Data = bootstrap_data[reweight_var][l_pt]['Central_Data'] * reweight_factor_pt_var

            p_Forward_Data = Normalize_unumpy(Forward_Data)
            p_Central_Data = Normalize_unumpy(Central_Data)

            extract_p_Quark_Data = f_inv[0][0] * p_Forward_Data + f_inv[0][1]* p_Central_Data 
            extract_p_Gluon_Data = f_inv[1][0] * p_Forward_Data + f_inv[1][1]* p_Central_Data 

            Extraction_Results_bootstrap[reweight_var][l_pt] = {
                    "Central_Data_event_weight": bootstrap_data[reweight_var][l_pt]['Central_Data'],
                    "Forward_Data": Forward_Data,
                    "Central_Data": Central_Data,
                    "f": Extraction_Results[reweight_var][l_pt]['f'].copy(),
                    "f_inv": f_inv,
                    "p_Quark": Extraction_Results[reweight_var][l_pt]['p_Quark'].copy(),
                    "p_Gluon": Extraction_Results[reweight_var][l_pt]['p_Gluon'].copy(),
                    "p_Forward_Quark": Extraction_Results[reweight_var][l_pt]['p_Forward_Quark'].copy(),
                    "p_Central_Quark": Extraction_Results[reweight_var][l_pt]['p_Central_Quark'].copy(),
                    "p_Forward_Gluon": Extraction_Results[reweight_var][l_pt]['p_Forward_Gluon'].copy(),
                    "p_Central_Gluon": Extraction_Results[reweight_var][l_pt]['p_Central_Gluon'].copy(),
                    "extract_p_Quark_MC": Extraction_Results[reweight_var][l_pt]['extract_p_Quark_MC'].copy(),
                    "extract_p_Gluon_MC": Extraction_Results[reweight_var][l_pt]['extract_p_Gluon_MC'].copy(),
                    "extract_p_Quark_Data": extract_p_Quark_Data,
                    "extract_p_Gluon_Data": extract_p_Gluon_Data
                }
    SFs = {}
    for var in SF_label_vars:
        reweighting_var = var 
        weight_option = "quark_reweighting_weights"
        period = "ADE"
        SFs[var] = {}
        WP_cut_path = nominal_path / "plots" / "ADE" / "WP_cuts_pkls" / f"{var}_quark_reweighting_weights" / "WP_cuts.pkl"
        WP_cut = joblib.load(WP_cut_path)
        #### Draw working points 
        for WP in WPs:
            SFs[var][WP] = {}
            quark_effs_at_pt = []
            gluon_rejs_at_pt = []
            quark_effs_data_at_pt = []
            gluon_rejs_data_at_pt = []
            for ii, l_pt in enumerate(label_pt_bin[:-1]):
                extract_p_Quark_MC =  Extraction_Results_bootstrap[var][l_pt]['extract_p_Quark_MC']
                extract_p_Gluon_MC =  Extraction_Results_bootstrap[var][l_pt]['extract_p_Gluon_MC']
                extract_p_Quark_Data =  Extraction_Results_bootstrap[var][l_pt]['extract_p_Quark_Data']
                extract_p_Gluon_Data =  Extraction_Results_bootstrap[var][l_pt]['extract_p_Gluon_Data']

                cut = WP_cut[var][WP][l_pt]['idx']
                

                # for others, compare MC extracted vs Data extracted after reweighting 
                quark_effs_at_pt.append(np.sum(extract_p_Quark_MC[:cut])) 
                gluon_rejs_at_pt.append(np.sum(extract_p_Gluon_MC[cut:]))
                quark_effs_data_at_pt.append(np.sum(extract_p_Quark_Data[:cut]))
                gluon_rejs_data_at_pt.append(np.sum(extract_p_Gluon_Data[cut:]))


            SF_quark, SF_gluon = Plot_WP(WP = WP, var= var, output_path= None, 
                    period= period, reweighting_var = reweighting_var,
                    reweighting_factor= weight_option,
                    quark_effs= quark_effs_at_pt, gluon_rejs = gluon_rejs_at_pt,
                    quark_effs_data=quark_effs_data_at_pt, gluon_rejs_data = gluon_rejs_data_at_pt,
                    if_save=False)
            SFs[var][WP]["Quark"] = SF_quark
            SFs[var][WP]["Gluon"] = SF_gluon
    return SFs

def bootstrap_once(nominal_path:Path, output_path:Path):
    data_Forward_Central_periods_path = output_path / "data_Forward_Central_periods.pkl"
    if not data_Forward_Central_periods_path.exists():
        data_Forward_Central_periods = generate_nominal(nominal_path)
        joblib.dump(data_Forward_Central_periods, data_Forward_Central_periods_path)
    else:
        data_Forward_Central_periods = joblib.load(data_Forward_Central_periods_path)

    bootstrapped_data = {}
    for k, v in data_Forward_Central_periods.items():
        bootstrapped_data[k] = bootstrap_data(v, period=k)

    bootstrapped_data = merged_bootstrap_period(bootstrapped_data)
    SFs = calculate_sf(bootstrapped_data)
    return SFs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nominal-path', help='the input folder path for MC', type=str, default=nominal_path)
    parser.add_argument('--output-path', help='the output folder path', type=str, default=output_path)

    args = parser.parse_args()
    nominal_path = Path(args.nominal_path)
    output_path = Path(args.output_path)
    bootstrap_once(nominal_path, output_path)
    
