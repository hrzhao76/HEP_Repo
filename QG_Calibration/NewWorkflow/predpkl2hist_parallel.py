import argparse
from pathlib import Path
from core.pkl2predpkl import * 
from concurrent.futures import ProcessPoolExecutor
import functools 
from core.utils import *

# default gbdt path 
reweight_file_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/test_reweight/test_reweight_factor2.pkl'
reweight_file_path = Path(reweight_file_path)

n_workers = 6

def predpkl2hist_parallel(input_path, do_reweight, reweight_factor_path, output_path = None, is_MC = True):
    input_path = check_inputpath(input_path)
    if do_reweight:
        reweight_factor_path = check_inputpath(reweight_factor_path)
        reweight_factor = joblib.load(reweight_factor_path)

    # if is_MC:
    #     period_list = ["A", "D", "E"]
    #     pkls_periods = [f"pythia{period}_predpkl" for period in period_list]
    # else:
    #     period_list = ["1516", "17", "18"]
    #     pkls_periods = [f"data{period}_pkl" for period in period_list]

    glob_pattern = "*_pred.pkl"
    merged_pkl_files = sorted(input_path.glob(glob_pattern))
    
    attach_reweight_factor_mod = functools.partial(attach_reweight_factor, reweight_factor=reweight_factor)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        reweighted_samples = list(executor.map(attach_reweight_factor_mod, merged_pkl_files))  

    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='the input folder path', type=str)
    parser.add_argument('--output-path', help='the output folder path', type=str)
    parser.add_argument('--do-reweight', help='flag if do reweighting', action='store_true')
    parser.add_argument('--reweight-file-path', help='the lightGBM model path', type=str, default=reweight_file_path)

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    predpkl2hist_parallel(input_path, do_reweight=args.do_reweight, reweight_factor_path=args.reweight_file_path,  
                          output_path=output_path, is_MC = True) 

    # predpkl2hist_parallel(input_path, do_reweight=args.do_reweight, reweight_factor_path = ar, 
    #                       output_path=output_path, is_MC = False)

