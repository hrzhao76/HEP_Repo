import argparse
from pathlib import Path
from pkl2predpkl import * 
from concurrent.futures import ProcessPoolExecutor
import functools 
from utils import check_inputpath, check_outputpath

# default gbdt path 
gbdt_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/LightGBM/optuna_tuning/small_dataset/lightgbm_gbdt.pkl'
gbdt_path = Path(gbdt_path)

n_workers = 8

def pkl2predpkl_parallel(input_path, gbdt_path = None, output_path = None, is_MC = True, if_save = False):
    input_path = check_inputpath(input_path)

    if is_MC:
        period_list = ["A", "D", "E"]
        pkls_periods = [f"pythia{period}_pkl" for period in period_list]
    else:
        period_list = ["1516", "17", "18"]
        pkls_periods = [f"data{period}_pkl" for period in period_list]

    glob_pattern = "*minitrees.pkl"

    for pkls_period in pkls_periods:
        pkls_path = input_path / pkls_period
        logging.info(f"Do inference on pkls from {pkls_path}...")
        predpkl_output_path = output_path / (pkls_period.split("_")[0] + "_predpkl") # forexample, pythiaA_predpkl 
        
        pkl_files = sorted(pkls_path.glob(glob_pattern))
        pkl2predpkl_mod = functools.partial(pkl2predpkl, output_path = predpkl_output_path, gbdt_path=gbdt_path, if_save=True)
            
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            pred_pkls_list = list(executor.map(pkl2predpkl_mod, pkl_files))  

        merged_pkls_period =  pd.concat(pred_pkls_list)
        merged_pkls_period_path = output_path / (pkls_period.split("_")[0] + "_pred.pkl") # forexample, pythiaA_pred.pkl
        joblib.dump(merged_pkls_period, merged_pkls_period_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='the input folder path', type=str)
    parser.add_argument('--output-path', help='the output folder path', type=str)
    parser.add_argument('--gbdt-path', help='the lightGBM model path', type=str, default=gbdt_path)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    pkl2predpkl_parallel(input_path, gbdt_path = gbdt_path, output_path=output_path, is_MC = True) 
    # uses 8 cores, 30 mins for pythia A, D, E, the merged file sizes are 6.4G, 4.6G and 13G 

    pkl2predpkl_parallel(input_path, gbdt_path = gbdt_path, output_path=output_path, is_MC = False)
    # uses 8 cores, 10 mins for data 1516, 17, 18, the merged file sizes are 2.2G, 2.6G and 3.5G 

