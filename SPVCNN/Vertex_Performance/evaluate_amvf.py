import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ROOT
import uproot 
import awkward as ak
import os
import re
from tqdm import tqdm

from core.utils import check_outputpath
from core.utils import vxMatchWeight, cutMinTruthRecoRadialDiff, VertexMatchType, HardScatterType

from core.differenceZ import performance_differenceZ
from core.num_vtx_vs_pu import performance_num_vtx_vs_pu
from core.classification_n_eff import performance_classification_n_eff

def evaluate_amvf(
    input_root_path:Path,
    output_path:Path,
    ):
    performance_differenceZ(input_root_path, output_path)
    performance_num_vtx_vs_pu(input_root_path, output_path)
    performance_classification_n_eff(input_root_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root-path', help='the input root file path', type=str)
    parser.add_argument('--output-path', help='the output path for performance plot', type=str)
    args = parser.parse_args()

    input_root_path = Path(args.input_root_path)
    output_path = args.output_path

    if output_path is None:
        output_path = input_root_path.parent
    else:
        output_path = Path(args.output_path)

    performance_classification_n_eff(input_root_path, output_path) 