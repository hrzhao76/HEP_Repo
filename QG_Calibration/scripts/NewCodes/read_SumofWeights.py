import argparse
import numpy as np

import uproot 
import awkward as ak
from pathlib import Path


def read_SumofWeights_Period(sample_folder_path, period):
    """Sum of the first value of unmerged hist.root files and return a numpy array. The length depends on the JZ slices. 

    Args:
        sample_folder_path (Path()): The Path to the rucio downloaded folders.
        period (String): Choose from ["A", "D", "E"], corresponding to periods. 

    Returns:
        Numpy Array: This array is the some of weights from different JZ slices. 
    """
    if not period in ["A", "D", "E"]:
        raise Exception(f'Period {period} not in supported periods. Currently supported: ["A", "D", "E"]')
    period_JZslice = sorted(sample_folder_path.rglob(f"*pythia{period}*mc16_13TeV.36470*.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ*WithSW_hist"))
    period_JZ_sum = np.zeros(len(period_JZslice), dtype= float)
    for i, dir in enumerate(period_JZslice):
        print(dir)
        sum_JZ_slice = 0 
        for file in sorted(dir.glob("*.hist-output.root")):
            sum_JZ_slice += uproot.open(file)['histoEventCount'].values()[0]
        
        period_JZ_sum[i] = sum_JZ_slice

    return period_JZ_sum



if __name__ == '__main__':
    """This main func is an example of calling read_SumWeights_Period() and save the sum of weights to the file. 
    """
    parser = argparse.ArgumentParser(description = 'This python script calculate pythia weights.')
    parser.add_argument('--path', help='The path to the hist files')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E'])
    args = parser.parse_args()

    
    period_JZ_sum = read_SumofWeights_Period(Path(args.path), args.period)

    print(period_JZ_sum)

    np.save(Path(args.path) / f'SumofWeights_mc16{args.period}', period_JZ_sum)
