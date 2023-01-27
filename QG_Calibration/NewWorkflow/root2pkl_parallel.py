import argparse
from pathlib import Path
from root2pkl import *
from concurrent.futures import ProcessPoolExecutor
import functools 
from utils import check_inputpath, check_outputpath

pythia_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/'
data_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_New/data/'
pythia_path = Path(pythia_path)
data_path = Path(data_path)

# def all_in_one():
#     #### Change some parameter here
#     root2pkl = functools.partial(root2pkl, output_path='./processed_pythia', verbosity=2, write_log=False)
#     root_files = []
#     for period in ["A", "D", "E"]:
#         pythia_path = f'/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/pythia{period}/'
#         pythia_path = Path(pythia_path)

#         root_files.append(sorted(pythia_path.rglob("*JZ?WithSW_minitrees.root/*.root")))

#     flat_list = list(np.concatenate(root_files).flat)

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         results = list(executor.map(root2pkl, flat_list))

def split_ADE(input_path, output_path = None, is_MC = True, verbosity = 2, write_log = False):

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

    for minitrees_period in minitrees_periods:

        minitreess = input_path / minitrees_period 
        root_files = sorted(minitreess.rglob(glob_pattern))

        pkl_output_path = output_path / (minitrees_period+"_pkl")

        root2pkl_mod = functools.partial(root2pkl, output_path=pkl_output_path, verbosity=verbosity, write_log=write_log)
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(root2pkl_mod, root_files)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mc-path', help='the output folder path', type=str, default=pythia_path)
    parser.add_argument('--input-data-path', help='the output folder path', type=str, default=data_path)
    parser.add_argument('--output-path', help='the output folder path', type=str)

    args = parser.parse_args()

    output_path = Path(args.output_path)
    input_mc_path = Path(args.input_mc_path)
    input_data_path = Path(args.input_data_path)
    # all_in_one()
    split_ADE(input_path = input_mc_path, output_path = output_path, is_MC = True)
    split_ADE(input_path = input_data_path, output_path = output_path, is_MC = False)
    # split_Data(output_path, is_MC = False)