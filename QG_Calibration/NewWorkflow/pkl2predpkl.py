import logging
import joblib 
import numpy as np
import pandas as pd 
from pathlib import Path
import re
from utils import check_inputpath, check_outputpath

def pkl2predpkl(pkl_path, gbdt_path, training_vars=None, output_path=None, if_save = True):

    pkl_path = check_inputpath(pkl_path)
    gbdt_path = check_inputpath(gbdt_path)

    if output_path is None:
        output_path = pkl_path.parent
    if training_vars is None:
        training_vars = ['jet_pt', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1']
    if if_save:
        output_path = check_outputpath(output_path)

    sample_pd = joblib.load(pkl_path)
    gbdt = joblib.load(gbdt_path)

    sample_pd['GBDT_newScore'] = gbdt.predict(sample_pd[training_vars], raw_score = True)
    if if_save:
        joblib.dump(sample_pd, output_path / (pkl_path.stem + "_pred.pkl"))
    return sample_pd
