import joblib 
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import hist
from hist import Hist 
from uncertainties import ufloat, unumpy
from utils import * 

def predpkl2hist(input, reweight='event_weight', output_path=None, if_save = False):
    if isinstance(input, pd.DataFrame):
        sample = input
    elif isinstance(input, str) or isinstance(input, Path):
        input_path = input if isinstance(input, Path) else Path(input)

        input_path = check_inputpath(input_path)
        sample = joblib.load(input_path)
        if not isinstance(input, pd.DataFrame):
            raise Exception(f"Check the input format! expect pd.DataFrame in {input_path}")
        
        if if_save and output_path is None:
            output_path = input_path.parent
            output_path = check_outputpath(output_path)
            output_name = f"digitized_{input_path.stem}.pkl"


    sample = sample[(sample["jet_nTracks"] > 1) & (sample["target"] != '-')] 
    sample = digitize_pd(sample, reweight=reweight)
    
    if if_save:
        joblib.dump(sample, output_path / output_name)


