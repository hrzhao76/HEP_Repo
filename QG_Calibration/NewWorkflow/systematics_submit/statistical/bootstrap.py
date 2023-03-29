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
import logging

from concurrent.futures import ProcessPoolExecutor
import functools 


import sys
core_code_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow'
sys.path.append(core_code_path)

from core.Calculate_SF import convert_histdict2unumpy, Construct_unumpy, Normalize_unumpy, Plot_WP, WriteSFtoPickle
from core.utils import HistBins, label_var, label_leadingtype, label_etaregion, label_pt_bin, label_jettype, label_jettype_data, WPs,reweighting_vars
from core.utils import logging_setup, make_empty_hist

nominal_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/nominal'
output_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/statistical/test/'
n_workers = 10

def construct_Forward_Central(sample_pred_hist_period_event_weight:dict, sample_type:str):
    """This function combines leading and subleading jets in each pt and central/forward regions. 

    Args:
        sample_pred_hist_period_event_weight (_type_): _description_

    Returns:
        _type_: _description_
    """
    Sample_period = dict.fromkeys(label_var)
    
    if sample_type == "Data":
        label_jettypes = label_jettype_data
    if sample_type == "MC":
        label_jettypes = label_jettype
    for var in label_var:
        Sample_period[var] = dict.fromkeys(label_pt_bin[:-1])
        for l_pt in label_pt_bin[:-1]:
            sel_HistMap_Sample_unumpy = {}
            for l_leadingtype  in label_leadingtype:
                for l_etaregion in label_etaregion:
                    for l_jettype in label_jettypes:

                        key_sample = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + l_jettype + "_" + var
                        sel_HistMap_Sample_unumpy[key_sample] = sample_pred_hist_period_event_weight[key_sample]
                        # if sample_type == "Data":
                        #     Forward_Sample, Central_Sample = \
                        #     Construct_unumpy(HistMap_unumpy=sel_HistMap_Sample_unumpy, n_bins = len(HistBins[var]) - 1, sampletype=sample_type)
                        # else:
                        Forward_Sample, Central_Sample, *others = \
                            Construct_unumpy(HistMap_unumpy=sel_HistMap_Sample_unumpy, n_bins = len(HistBins[var]) - 1, sampletype=sample_type)

            Sample_period[var][l_pt] = {
                f"Forward_{sample_type}": Forward_Sample,
                f"Central_{sample_type}": Central_Sample,
            } 
    
    return Sample_period

def bootstrap_sample(nominal_Sample, period, sample_type:str):
    bootstrap_Sample = dict.fromkeys(nominal_Sample.keys())
    
    for var, sample_var in nominal_Sample.items():
        bootstrap_Sample[var] = dict.fromkeys(sample_var.keys())
        for pt, sample_pt in sample_var.items():
            bootstrap_Sample[var][pt] = dict.fromkeys(sample_pt.keys())
            for region, sample_region in sample_pt.items():
                hist_var_pt_region_unumpy = unumpy.nominal_values(sample_region)

                if sample_type == "Data":
                    bootstrap = np.random.poisson(lam=hist_var_pt_region_unumpy)
                    if period == "18":
                        variances = (58.45/39.91) ** 2 * bootstrap # Note the precedence of the operator `**` 
                    else: 
                        variances = bootstrap

                if sample_type == "MC":
                    bootstrap = np.random.normal(loc=hist_var_pt_region_unumpy, scale=unumpy.std_devs(sample_region))
                    variances = unumpy.std_devs(sample_region) ** 2  # For MC bootstrapping, the bin errors are unchanged. 

                bootstrap_Sample[var][pt][region] =  unumpy.uarray(bootstrap, np.sqrt(variances))
    return bootstrap_Sample

def merged_bootstrap_period(bootstrap_sample):
    keys = [*bootstrap_sample.keys()]
    
    bootstrap_sample_merged = bootstrap_sample[keys[0]].copy()

    for bootstrap_sample_key in keys[1:]:
        bootstrap_Sample_period = bootstrap_sample[bootstrap_sample_key]
        for var, sample_var in bootstrap_Sample_period.items():
            for pt, sample_pt in sample_var.items():
                for region, sample_region in sample_pt.items():
                    bootstrap_sample_merged[var][pt][region] += sample_region
    
    return bootstrap_sample_merged

def generate_nominal(nominal_path:Path, sample_type:str):
    if not sample_type in ["MC", "Data"]:
        raise NotImplementedError()

    if sample_type == "Data":
        type_period = ['1516', '17', '18'] 
        type_pattern = "data*_pred_hists.pkl"
        type_re_pattern = r"data(\d+)_pred_hists"
    elif sample_type == "MC":
        type_period = ['A', 'D', 'E_part1', 'E_part2'] 
        type_pattern = "pythia*_pred_hists.pkl"
        type_re_pattern = r"pythia(\w+)_pred_hists"

    sample_Forward_Central_periods = dict.fromkeys(type_period)

    for file in sorted(nominal_path.rglob(type_pattern)):
        period_search_pattern = type_re_pattern
        period = re.search(period_search_pattern, file.stem).group((1))
        assert period in type_period
        
        sample_pred_hist_period = joblib.load(file)
        sample_pred_hist_period_event_weight = convert_histdict2unumpy(sample_pred_hist_period['event_weight']) 
        sample_Forward_Central_periods[period] = construct_Forward_Central(sample_pred_hist_period_event_weight, sample_type)

    return sample_Forward_Central_periods

def calculate_sf(bootstrapped_Samples:tuple, nominal_path:Path):
    bootstrapped_MC =  bootstrapped_Samples[0]
    bootstrapped_Data = bootstrapped_Samples[1]

    reweight_factor_path = nominal_path / "reweight_factor.pkl"
    reweight_factor = joblib.load(reweight_factor_path)

    Extraction_Results_bootstrap = {}

    SF_label_vars = ['jet_nTracks', 'GBDT_newScore']
    
    for reweight_var in SF_label_vars:

        Extraction_Results_bootstrap[reweight_var] = {}
        # Load information from MC 
        Extraction_Results_path = nominal_path / "plots" / "ADE" / "Extraction_Results" / f"{reweight_var}_Extraction_Results.pkl"
        Extraction_Results = joblib.load(Extraction_Results_path)

        for l_pt in label_pt_bin[:-1]:
            reweight_factor_pt_var = reweight_factor[l_pt][reweight_var]['quark_factor']

            f_inv = Extraction_Results[reweight_var][l_pt]['f_inv']
            
            Forward_MC = bootstrapped_MC[reweight_var][l_pt]['Forward_MC']
            Central_MC = bootstrapped_MC[reweight_var][l_pt]['Central_MC'] * reweight_factor_pt_var

            Forward_Data = bootstrapped_Data[reweight_var][l_pt]['Forward_Data']
            Central_Data = bootstrapped_Data[reweight_var][l_pt]['Central_Data'] * reweight_factor_pt_var

            p_Forward_MC = Normalize_unumpy(Forward_MC)
            p_Central_MC = Normalize_unumpy(Central_MC)
            p_Forward_Data = Normalize_unumpy(Forward_Data)
            p_Central_Data = Normalize_unumpy(Central_Data)


            extract_p_Quark_MC = f_inv[0][0] * p_Forward_MC + f_inv[0][1]* p_Central_MC 
            extract_p_Gluon_MC = f_inv[1][0] * p_Forward_MC + f_inv[1][1]* p_Central_MC 

            extract_p_Quark_Data = f_inv[0][0] * p_Forward_Data + f_inv[0][1]* p_Central_Data 
            extract_p_Gluon_Data = f_inv[1][0] * p_Forward_Data + f_inv[1][1]* p_Central_Data 

            Extraction_Results_bootstrap[reweight_var][l_pt] = {
                    "Central_Data_bootstrap_event_weight": bootstrapped_Data[reweight_var][l_pt]['Central_Data'],
                    "Central_MC_bootstrap_event_weight": bootstrapped_MC[reweight_var][l_pt]['Central_MC'],
                    "Forward_Data": Forward_Data,
                    "Central_Data": Central_Data,
                    "f": Extraction_Results[reweight_var][l_pt]['f'].copy(),
                    "f_inv": f_inv,
                    # "p_Quark": Extraction_Results[reweight_var][l_pt]['p_Quark'].copy(),
                    # "p_Gluon": Extraction_Results[reweight_var][l_pt]['p_Gluon'].copy(),
                    # "p_Forward_Quark": Extraction_Results[reweight_var][l_pt]['p_Forward_Quark'].copy(),
                    # "p_Central_Quark": Extraction_Results[reweight_var][l_pt]['p_Central_Quark'].copy(),
                    # "p_Forward_Gluon": Extraction_Results[reweight_var][l_pt]['p_Forward_Gluon'].copy(),
                    # "p_Central_Gluon": Extraction_Results[reweight_var][l_pt]['p_Central_Gluon'].copy(),
                    "extract_p_Quark_MC": extract_p_Quark_MC,
                    "extract_p_Gluon_MC": extract_p_Gluon_MC,
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
            for l_pt in label_pt_bin[:-1]:
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

def convert_SFslist2SFspd(SFs_all_list:list):
    SFs_all_pd = \
    pd.DataFrame.from_records(
        [
            (level1, level2, level3, level4, leaf_idx, unumpy.nominal_values(leaf_pt)) # The sizes of unumpy is huge, here just the nominal_values are stored. 
            for level1, level2_dict in enumerate(SFs_all_list)
            for level2, level3_dict in level2_dict.items()
            for level3, level4_dict in level3_dict.items()
            for level4, leaf in level4_dict.items()
            for leaf_idx, leaf_pt in enumerate(leaf)
        ],
        columns=['Trail', 'Var', 'WP', 'Reweight', 'pt_idx', 'values']
    )
    return SFs_all_pd

def check_Forward_Central_exist(Sample_Forward_Central_periods_path:Path, nominal_path:Path, sample_type:str):
    if not Sample_Forward_Central_periods_path.exists():
        sample_Forward_Central_periods = generate_nominal(nominal_path, sample_type)
        Sample_Forward_Central_periods_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(sample_Forward_Central_periods, Sample_Forward_Central_periods_path)
    else:
        sample_Forward_Central_periods = joblib.load(Sample_Forward_Central_periods_path)

    return sample_Forward_Central_periods 

def convert_SFspd2SFshist(SFs_all_pd):
    
    Merged_Hist = dict.fromkeys(reweighting_vars)

    partons = ['Quark', 'Gluon']
    for var in reweighting_vars:
        Merged_Hist[var] = dict.fromkeys(WPs)
        for WP in WPs:
            Merged_Hist[var][WP] = dict.fromkeys(partons)
            for parton in partons:
                Merged_Hist[var][WP][parton] = dict.fromkeys(label_pt_bin[:-1])
                for pt_idx, l_pt in enumerate(label_pt_bin[:-1]):
                    values_unumpy = SFs_all_pd.loc[(SFs_all_pd['Var'] == var) & (SFs_all_pd['WP'] == WP) 
                                                    & (SFs_all_pd['Reweight'] == parton) & ((SFs_all_pd['pt_idx'] == pt_idx)),
                                                    'values']
                    hist_holder = make_empty_hist(bins=np.linspace(0.95, 1.05, 101))
                    hist_holder.fill(unumpy.nominal_values(values_unumpy))
                    Merged_Hist[var][WP][parton][l_pt] = hist_holder
    
    return Merged_Hist

def bootstrap(Sample_Forward_Central_periods:dict, sample_type:str):
    bootstrapped_Sample_periods = {}

    for k, v in Sample_Forward_Central_periods.items():
        bootstrapped_Sample_periods[k] = bootstrap_sample(nominal_Sample=v, period=k, sample_type=sample_type)

    bootstrapped_Sample = merged_bootstrap_period(bootstrapped_Sample_periods)

    return bootstrapped_Sample

def bootstrap_once(nominal_path:Path, output_path:Path):
    Data_Forward_Central_periods_path = output_path / "Data_Forward_Central_periods.pkl"
    Data_Forward_Central_periods = check_Forward_Central_exist(Data_Forward_Central_periods_path, nominal_path, sample_type="Data")
    MC_Forward_Central_periods_path = output_path / "MC_Forward_Central_periods.pkl"
    MC_Forward_Central_periods = check_Forward_Central_exist(MC_Forward_Central_periods_path, nominal_path, sample_type="MC")

    bootstrapped_Data = bootstrap(Data_Forward_Central_periods, "Data")
    bootstrapped_MC = bootstrap(MC_Forward_Central_periods, "MC")

    bootstrapped_version_SF = calculate_sf(nominal_path, (bootstrapped_MC, bootstrapped_Data))
    # For test case 
    # joblib.dump(bootstrapped_Data, output_path/"bootstrapped_Data.pkl")
    # joblib.dump(bootstrapped_MC, output_path/"bootstrapped_MC.pkl")
    # joblib.dump(bootstrapped_version_SF, output_path/"bootstrapped_version_SF.pkl")
    return bootstrapped_version_SF

def generate_bootstrap_samples(nominal_path:Path, if_save=False, output_path=None):
    Data_Forward_Central_periods_path = output_path / "Data_Forward_Central_periods.pkl"
    Data_Forward_Central_periods = check_Forward_Central_exist(Data_Forward_Central_periods_path, nominal_path, sample_type="Data")
    MC_Forward_Central_periods_path = output_path / "MC_Forward_Central_periods.pkl"
    MC_Forward_Central_periods = check_Forward_Central_exist(MC_Forward_Central_periods_path, nominal_path, sample_type="MC")
    
    # for i in tqdm(range(50)):
    bootstrapped_samples = []
    for j in range(100):
        bootstrapped_Data = bootstrap(Data_Forward_Central_periods, "Data")
        bootstrapped_MC = bootstrap(MC_Forward_Central_periods, "MC")

        bootstrapped_samples.append((bootstrapped_MC, bootstrapped_Data))
    
    if if_save:
        joblib.dump(bootstrapped_samples, output_path / f"bootstrapped_samples")

    return bootstrapped_samples

def bootstrap_parallel(input_file:Path, output_path:Path, output_name="SFs_all.pkl"):
    logging_setup(verbosity=3, if_write_log=False, output_path=output_path)
    logging.info("Doing bootstrap in parallel...")
    logging.info(f"The input file is {input_file}")
    bootstrapped_samples = joblib.load(input_file)

    # bootstrapped_samples = generate_bootstrap_samples(nominal_path, if_save=False, output_path=output_name)

    calculate_sf_mod = functools.partial(calculate_sf, nominal_path=nominal_path)

    logging.info("Starting parallel...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        SFs_all_list = list(executor.map(calculate_sf_mod, bootstrapped_samples))
    logging.info("Parallel done. Covert the list to pandas.")
    SFs_all_pd = convert_SFslist2SFspd(SFs_all_list)

    logging.info("pandas done. Covert the dataframe to hist.")
    SFs_all_hist = convert_SFspd2SFshist(SFs_all_pd)

    output_name = input_file.stem

    output_file_path = output_path / f"SFs_pd_{output_name}"
    logging.info(f"Writing out the SFs Dataframe... \n{output_file_path}")
    joblib.dump(SFs_all_pd, output_file_path)

    # output_file_path = output_path / f"SFs_hist_{output_name}"
    # logging.info(f"Writing out the SFs Hist... \n{output_file_path}")
    # joblib.dump(SFs_all_hist, output_file_path)

    logging.info(f"Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nominal-path', help='the input folder path for MC', type=str, default=nominal_path)
    parser.add_argument('--output-path', help='the output folder path', type=str, default=output_path)
    parser.add_argument('--mode', help='the mode of this file', type=str, choices=["test", "generation", "parallel"],default=output_path)
    parser.add_argument('--input-file', help='the ', type=str, default=None)

    args = parser.parse_args()
    nominal_path = Path(args.nominal_path)
    output_path = Path(args.output_path)
    mode = args.mode

    if mode == "test":
        bootstrap_once(nominal_path, output_path)
    if mode == "generation":
        generate_bootstrap_samples(nominal_path, output_path)
    if mode == "parallel":
        input_file = Path(args.input_file )
        # output_name = args.output_name
        bootstrap_parallel(input_file, output_path)
    # bootstrap_parallel(nominal_path, output_path)
    
