import numpy as np
import pandas as pd
from pathlib import Path
import hist
from hist import Hist
from uncertainties import ufloat, unumpy

###########################################################
###################Some global variables###################
###########################################################

label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]
label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt", "newBDT"]
label_leadingtype = ["LeadingJet", "SubLeadingJet"]
label_etaregion = ["Forward", "Central"]
label_jettype = ["Gluon", "Quark", "B_Quark", "C_Quark"]
label_jettype_data = ["Data"]

label_var_map = {
    'pt':'jet_pt',
    'eta':'jet_eta',
    'ntrk':'jet_nTracks', 
    'width':'jet_trackWidth',
    'c1':'jet_trackC1', 
    'bdt':'jet_trackBDT', 
    'newBDT':'GBDT_newScore'
}

is_leading_map = {
    "LeadingJet": [1],
    "SubLeadingJet": [0],
}

is_forward_map = {
    "Forward": [1],
    "Central": [0],
}

label_jettype_map = {
    "Gluon" : [21], 
    "Quark" : [1, 2, 3],
    "B_Quark" : [5],
    "C_Quark" : [4],
    "Data" : [-9999.0]
}

HistBins = {
    'jet_pt' : np.linspace(500, 2000, 61),
    'jet_eta' : np.linspace(-2.5, 2.5, 51), 
    'jet_nTracks' : np.linspace(0, 60, 61),
    'jet_trackWidth' : np.linspace(0, 0.4, 61),
    'jet_trackC1' : np.linspace(0, 0.4, 61),
    'jet_trackBDT' : np.linspace(-1.0, 1.0, 101),
    'GBDT_newScore' : np.linspace(-5.0, 5.0, 101),
}

###########################################################
###################Some useful functions###################
###########################################################
def check_inputpath(input_path):
    if not isinstance(input_path, Path):
        input_path = Path(input_path)
    if not input_path.exists():
        raise Exception(f"File {input_path} not found. ")
    return input_path

def check_outputpath(output_path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path 

def normalize_hist(_hist):
    area = np.sum(_hist.values()) * _hist.axes[0].widths
    return _hist / area
    

def make_hist(values, bins, weights):
    # assuming bins numpy array with (start, stop, n_edges)
    _hist = Hist(hist.axis.Regular(bins=len(bins)-1, start=bins[0], stop=bins[-1], overflow=True, underflow=True), 
                                storage=hist.storage.Weight())
    _hist.fill(values, weight=weights)
    _normed_hist = normalize_hist(_hist)

    return _hist, _normed_hist

def digitize_pd(pd_input, reweight='event_weight', data_type = 'MC'):

    HistMap_unumpy = {}
    if data_type == 'MC':
        _label_jettype = label_jettype
    elif data_type == 'Data':
        _label_jettype = label_jettype_data
    for pt_idx, pt in enumerate(label_pt_bin[:-1]):
        pt_input_idx = pd_input['pt_idx'] == pt_idx
        pd_input_at_pt = pd_input[pt_input_idx]

        for leadingtype in label_leadingtype:
            leadingtype_idx = pd_input_at_pt['is_leading'].isin(is_leading_map[leadingtype])
            pd_input_at_pt_leadingtype = pd_input_at_pt[leadingtype_idx]
            
            for eta_region in label_etaregion: 
                etaregion_idx = pd_input_at_pt_leadingtype['is_forward'].isin(is_forward_map[eta_region])
                pd_input_at_pt_leadingtype_etaregion = pd_input_at_pt_leadingtype[etaregion_idx]
                
                for jettype in _label_jettype:
                    type_idx = pd_input_at_pt_leadingtype_etaregion['jet_PartonTruthLabelID'].isin(label_jettype_map[jettype])
                    pd_input_at_pt_leadingtype_etaregion_jettype = pd_input_at_pt_leadingtype_etaregion[type_idx]
                    for var in label_var:
                        key = f"{pt}_{leadingtype}_{eta_region}_{jettype}_{var}"
                        bin_var = HistBins[label_var_map[var]]

                        # TODO: Can change the format from unumpy to hist. Now just to test the plotting code. 
                        if len(pd_input_at_pt_leadingtype_etaregion_jettype) == 0: ## for subset, if len==0, give it an empty unumpy array
                            HistMap_unumpy[key] = unumpy.uarray(np.zeros(len(bin_var)-1), np.zeros(len(bin_var)-1))
                            continue
                        else:
                            _hist, _norm_hist = make_hist(values=pd_input_at_pt_leadingtype_etaregion_jettype[label_var_map[var]],
                                                bins=bin_var, weights=pd_input_at_pt_leadingtype_etaregion_jettype[reweight])
                            HistMap_unumpy[key] = unumpy.uarray(_hist.values(), np.sqrt(_hist.variances()))
                            
    return HistMap_unumpy

def attach_reweight_factor(pd_input, reweight_factor):
    reweighting_vars = ['jet_nTracks', 'jet_trackBDT', 'GBDT_newScore'] 
    for reweighting_var in reweighting_vars:
        pd_input[f'{reweighting_var}_quark_reweighting_weights'] = pd_input['event_weight'].copy()
        pd_input[f'{reweighting_var}_gluon_reweighting_weights'] = pd_input['event_weight'].copy()

    reweighted_sample = []
    #### reweight_factor[pt][var]['quark_factor']
    for pt_idx, pt in enumerate(label_pt_bin[:-1]):
        pd_input_at_pt = pd_input[pd_input['pt_idx'] == pt_idx]
        pd_input_at_pt_forward = pd_input_at_pt[pd_input_at_pt['is_forward']==1]
        pd_input_at_pt_central = pd_input_at_pt[pd_input_at_pt['is_forward']==0]

        for reweighting_var in reweighting_vars:
            bin_var = HistBins[reweighting_var]
            quark_factor_idx = pd_input_at_pt.columns.get_loc(f'{reweighting_var}_quark_reweighting_weights')
            gluon_factor_idx = pd_input_at_pt.columns.get_loc(f'{reweighting_var}_gluon_reweighting_weights')

            quark_factor = reweight_factor[pt][reweighting_var]['quark_factor']
            gluon_factor = reweight_factor[pt][reweighting_var]['gluon_factor']

            var_idx = np.digitize(x=pd_input_at_pt_central[reweighting_var] , bins=bin_var) - 1  # Binned feature distribution 
            for i, score in enumerate(bin_var[:-1]): # Loop over the bins 
                mod_idx = np.where(var_idx == i)[0]
                pd_input_at_pt_central.iloc[mod_idx, quark_factor_idx] *= quark_factor[i]
                pd_input_at_pt_central.iloc[mod_idx, gluon_factor_idx] *= gluon_factor[i]
            
        reweighted_sample.append(pd_input_at_pt_forward)
        reweighted_sample.append(pd_input_at_pt_central)                
    
    return pd.concat(reweighted_sample)

