from core.utils import *
from core.root2pkl import * 
from core.pkl2predpkl import *
from core.predpkl2hist import * 

from concurrent.futures import ProcessPoolExecutor
import functools 

pythia_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/'
data_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_New/data/'
pythia_path = Path(pythia_path)
data_path = Path(data_path)

# default gbdt path 
gbdt_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/LightGBM/optuna_tuning/small_dataset/lightgbm_gbdt.pkl'
gbdt_path = Path(gbdt_path)

n_workers = 8 


def root2hist(input_path, output_path = None, is_MC = True, verbosity = 2, write_log = False):
    input_path = check_inputpath(input_path)

    if is_MC:
        period_list = ["A", "D", "E"]
        minitrees_periods = [f"pythia{period}" for period in period_list]
        glob_pattern = "*JZ?WithSW_minitrees.root/*.root"
    else:
        period_list = ["1516", "17", "18"]
        minitrees_periods = [f"data{period}" for period in period_list]
        glob_pattern = "*data*_13TeV.period?.physics_Main_minitrees.root/*.root"
        
    if output_path is None:
            output_path = input_path
    else:
        output_path = check_outputpath(output_path)

    return_dicts = {}
    for minitrees_period in minitrees_periods:

        minitreess = input_path / minitrees_period 
        root_files = sorted(minitreess.rglob(glob_pattern))

        root2pkl_mod = functools.partial(root2pkl, output_path= None, verbosity=verbosity, write_log=write_log, if_save=False)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            pkls = list(executor.map(root2pkl_mod, root_files))
        
        pkls = [x for x in pkls if x is not None] # Important to filter the None! 
        
        pkl2predpkl_mod = functools.partial(pkl2predpkl, output_path = None, gbdt_path=gbdt_path, if_save=False)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            predpkls = list(executor.map(pkl2predpkl_mod, pkls))  

        predpkl2hist_mod = functools.partial(predpkl2hist, reweight='event_weight',is_MC = is_MC, output_path = None, if_save = False)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            hists = list(executor.map(predpkl2hist_mod, predpkls)) 

        merged_pkls_period = pd.concat(predpkls)
        merged_pkls_period_path = output_path / (f"{minitrees_period}_pred.pkl") # forexample, pythiaA_pred.pkl
        joblib.dump(merged_pkls_period, merged_pkls_period_path)

        return_dicts[minitrees_period]=hists

    return return_dicts

def final_reweighting(pkl:Path, reweight_factor, output_path = None):
    sample_pd = joblib.load(pkl)
    sample_pd = attach_reweight_factor(sample_pd, reweight_factor)
    if output_path is None:
        output_path = pkl.parent

    joblib.dump(sample_pd, pkl) # overwrite the original file 

    reweighted_hists_dict = {}
    is_MC = True if pkl.stem.__contains__('data') else "MC"
    for weight in all_weight_options:
        reweighted_hists_dict[weight] = digitize_pd(sample_pd, weight, is_MC=is_MC)

    joblib.dump(reweighted_hists_dict, output_path / f"{pkl.stem}_hists.pkl" )

    if is_MC:
        return_key = 'MC'
    else:
        return_key = 'Data'

    return {return_key: reweighted_hists_dict}

def _merge_period(hist_list):
    merged_hist = hist_list[0]
    for to_be_merged_hist in hist_list[1:]:
        for k, v in to_be_merged_hist:
            merged_hist[k] += v
    return merged_hist

def merge_period(reweighted_hists_dicts:list):
    # reweighted_hists_dicts is a list of dicts.
    # e.g. 6 dicts, 3 'MC' and 3 'Data'
    MC_hist_list = []
    Data_hist_list = []
    for reweighted_hists_dict in reweighted_hists_dicts:
        if [*reweighted_hists_dict][0] == 'MC':
            MC_hist_list.append(reweighted_hists_dict)
        elif [*reweighted_hists_dict][0] == 'Data':
            Data_hist_list.append(reweighted_hists_dict)
    
    return _merge_period(MC_hist_list), _merge_period(Data_hist_list)

def make_histogram_parallel(input_mc_path, input_data_path, output_path):
    logging_setup(verbosity=3, if_write_log=False, output_path=output_path)
    logging.info("Doing root2hist for MC...")
    MC_hists = root2hist(input_path=input_mc_path, output_path=output_path, is_MC=True)
    # breakpoint()
    # joblib.dump(MC_hists, output_path / 'MC_hists.pkl')
    # MC_hists = joblib.load(output_path / 'MC_hists.pkl')
    logging.info("Calculate reweighting factor from MC...")
    reweight_factor = get_reweight_factor_hist(MC_hists, if_need_merge=True)
    joblib.dump(reweight_factor, output_path / 'reweight_factor.pkl')
    # reweight_factor = joblib.load(output_path / 'reweight_factor.pkl')

    Data_hists =  root2hist(input_path=input_data_path, output_path=output_path, is_MC=False)
    predpkl_pattern = "*_pred.pkl"
    predpkl_files = sorted(output_path.rglob(predpkl_pattern))

    logging.info("Attach new weighting to pd.DataFrame and reweighting...")
    final_reweighting_mod = functools.partial(final_reweighting, reweight_factor = reweight_factor, output_path=output_path)
    with ProcessPoolExecutor(max_workers=6) as executor:
        reweighted_hists_dicts = list(executor.map(final_reweighting_mod, predpkl_files))
    
    breakpoint()
    logging.info("Merging the histograms for MC and Data...")
    MC_merged_hist, Data_merged_hist = merge_period(reweighted_hists_dicts)
    joblib.dump(MC_merged_hist, output_path / 'MC_merged_hist.pkl')
    joblib.dump(Data_merged_hist, output_path / 'Data_merged_hist.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mc-path', help='the output folder path', type=str, default=pythia_path)
    parser.add_argument('--input-data-path', help='the output folder path', type=str, default=data_path)
    parser.add_argument('--output-path', help='the output folder path', type=str)
    parser.add_argument('--gbdt-path', help='the lightGBM model path', type=str, default=gbdt_path)

    args = parser.parse_args()

    output_path = Path(args.output_path)
    input_mc_path = Path(args.input_mc_path)
    input_data_path = Path(args.input_data_path)

    make_histogram_parallel(input_mc_path=input_mc_path, input_data_path=input_data_path,
                            output_path=output_path)
    # all_in_one()
    # make_histogram_parallel(input_path = input_mc_path, output_path = output_path, is_MC = True)
    # make_histogram_parallel(input_path = input_data_path, output_path = output_path, is_MC = False)
    # split_Data(output_path, is_MC = False)
