import argparse
import uproot
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from pathlib import Path

def Construct(HistMap):
    ## Construct data-like MC 
    Forward = np.zeros((60))
    Central = np.zeros((60))
    for k, v in HistMap.items():
        if k.__contains__('Forward'):
            Forward += v[0]
        elif k.__contains__('Central'):
            Central += v[0]

    ## Construct pure Quark vs Gluon 
    Quark = np.zeros((60))
    Gluon = np.zeros((60))
    for k, v in HistMap.items():
        if k.__contains__('Quark'):
            Quark += v[0]
        elif k.__contains__('Gluon'):
            Gluon += v[0]

    Forward_Quark = np.zeros((60))
    Forward_Gluon = np.zeros((60))
    Central_Quark = np.zeros((60))
    Central_Gluon = np.zeros((60))

    for k, v in HistMap.items():
        if k.__contains__('Quark') and k.__contains__('Forward'):
            Forward_Quark += v[0]
        elif k.__contains__('Gluon') and k.__contains__('Forward'):
            Forward_Gluon += v[0]
        elif k.__contains__('Quark') and k.__contains__('Central'):
            Central_Quark += v[0]
        elif k.__contains__('Gluon') and k.__contains__('Central'):
            Central_Gluon += v[0]
    return Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon 

def Calcu_Frac(Forward_Quark, Central_Quark, Forward, Central):
    frac_Forward_Quark = np.sum(Forward_Quark) / np.sum(Forward)
    frac_Central_Quark = np.sum(Central_Quark) / np.sum(Central)

    frac_Forward_Gluon = 1 - frac_Forward_Quark
    frac_Central_Gluon = 1 - frac_Central_Quark

    f = np.array([[frac_Forward_Quark,  frac_Forward_Gluon], [frac_Central_Quark, frac_Central_Gluon]])

    return f, np.linalg.inv(f)

def MCclosure(input_path, period, reweighting_option, output_path):
    # define some variables to make up the histogram names 
    label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]
    label_var = ['ntrk', 'bdt'] 
    label_pt = ["LeadingJet", "SubLeadingJet"]
    label_eta = ["Forward", "Central"]
    label_type = ["Gluon", "Quark", "B_Quark", "C_Quark"]

    # defile which TDirectory to look at based on {reweighting_option}
    reweighting_map = {
        "No" : "NoReweighting",
        "quark" : "Reweighting_Quark_Factor",
        "gluon" : "Reweighting_Gluon_Factor"
    }
    mcfile = uproot.open(input_path)[reweighting_map[reweighting_option]]

    for var in label_var:
        for pt in label_pt_bin[:-1]:
            HistMap = {}

            for i, l_pt  in enumerate(label_pt):
                for j, l_eta in enumerate(label_eta):
                    for k, l_type in enumerate(label_type):
                        key = str(pt) + "_" + l_pt + "_" + l_eta + "_" + l_type + "_" + var
                        HistMap[key] = mcfile[key].to_numpy()

            Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct(HistMap=HistMap)

            f, f_inv = Calcu_Frac(Forward_Quark, Central_Quark, Forward, Central)

            # normalize 
            p_Quark = Quark / np.sum(Quark)
            p_Gluon = Gluon / np.sum(Gluon)

            p_Forward = Forward / np.sum(Forward)
            p_Central = Central / np.sum(Central)

            p_Forward_Quark = Forward_Quark / np.sum(Forward_Quark)
            p_Central_Quark = Central_Quark / np.sum(Central_Quark)
            p_Forward_Gluon = Forward_Gluon / np.sum(Forward_Gluon)
            p_Central_Gluon = Central_Gluon / np.sum(Central_Gluon)


            extract_p_Quark = f_inv[0][0] * p_Forward + f_inv[0][1]* p_Central 
            extract_p_Gluon = f_inv[1][0] * p_Forward + f_inv[1][1]* p_Central 

            

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script does the MC Closure test. ')
    parser.add_argument('--path', help='The path to the histogram file(.root file).')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E', "ADE"])
    parser.add_argument('--reweighting', help='The reweighting method', choices=['No', 'quark', 'gluon'])
    parser.add_argument('--output_path', help='Output path')
    args = parser.parse_args()

    root_file_path = Path(args.path)
    output_path = Path(args.output_path)
    period = args.period

    if root_file_path.suffix != ".root" :
        raise Exception(f"The input file {root_file_path} is not a root file! ")

    if period !=  root_file_path.stem[-len(period):]:
        raise Exception(f"The input file {root_file_path.stem} is not consistent with the period {period}!")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    MCclosure(input_path=root_file_path, period = period, reweighting_option = args.reweighting , output_path = output_path)
    

