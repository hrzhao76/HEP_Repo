import uproot
import argparse
import awkward as ak
import hist
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import re
import ROOT

from utils import check_outputpath

def performance_num_vtx_vs_pu(
        input_root_path:Path,
        output_path:Path,
        reco_tree_name='Reco_Vertex',
        reco_vtz_vz_name='reco_vtx_vz',
        truth_tree_name='Truth_Vertex_PV_Selected',
        truth_vtx_vz_name='truth_vtx_vz'):
    
    output_path = output_path / "num_vtx_vs_pu"
    check_outputpath(output_path)
    ### Reading the information from TTree
    root_file =uproot.open(input_root_path)
    file_name = input_root_path.stem # for example, vertexperformance_AMVF_pu100.root
    pu_search_pattern = r'pu(.+)$'
    pu_number = re.search(pu_search_pattern, file_name).group((1))

    reco_tree = root_file[reco_tree_name]
    reco_vtx_vz = reco_tree[reco_vtz_vz_name].array(library="ak")
    truth_tree = root_file[truth_tree_name]
    truth_vtx_vz = truth_tree[truth_vtx_vz_name].array(library="ak")

    num_truth_vtx_vz_events = root_file['amvf'].arrays()['nTrueVtx']
    # num_truth_vtx_vz_events = ak.count(truth_vtx_vz, axis=1)
    num_reco_vtx_vz_events = ak.count(reco_vtx_vz,axis=1)

    max_value = np.max(num_truth_vtx_vz_events)
    ### Plot the 
    TH2_vtx_vs_pu = (
        hist.Hist.new.Reg(max_value, 0, max_value, name="num_PU", label="Number of PU", flow=False)
        .Reg(max_value, 0, max_value, name="num_reco_vtx", label="Number of reco vertex", flow=False)
        .Double()
    )
    TH2_vtx_vs_pu.fill(num_truth_vtx_vz_events, num_reco_vtx_vz_events)
    hprofile = TH2_vtx_vs_pu.profile("num_reco_vtx")

    fig, ax = plt.subplots()
    hprofile.plot()
    # bin_values, bin_edges = hprofile.to_numpy()
    # ax.stairs(bin_values,  bin_edges)
    plt.ylabel("Number of Reconstructed Vertex")
    plt.savefig(output_path / "num_vtx_vs_pu.png")

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

    performance_num_vtx_vs_pu(input_root_path, output_path)

