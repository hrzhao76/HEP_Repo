import uproot
import argparse
import awkward as ak
import hist
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


def calculate_differenceZ(
        reco_vtx_vz : ak.highlevel.Array
        ):
    
    differenceZ = []
    for reco_vtx_vz_event in reco_vtx_vz:
        distance_all_pairs = ak.flatten(reco_vtx_vz_event[:, None] - reco_vtx_vz_event)
        differenceZ.append(distance_all_pairs[distance_all_pairs!=0])
    
    return differenceZ

def plot_hist(_hist:hist.hist.Hist, output_path=None, output_name=None):
    fig, ax = plt.subplots()
    bin_contents, bin_edges = _hist.to_numpy()
    ax.stairs(values=bin_contents, edges=bin_edges)
    ax.set_ylabel('Number of Vertices')
    ax.set_xlabel(r'$\Delta Z[mm]$')
    fig.savefig('test')

def performance_differenceZ(
        input_root_path:Path,
        reco_tree_name='Reco_Vertex',
        reco_vtz_vz_name='reco_vtx_vz'):
    
    root_file =uproot.open(input_root_path)
    reco_tree = root_file[reco_tree_name]
    reco_vtx_vz = reco_tree[reco_vtz_vz_name].array(library="ak")

    hist_differenceZ = hist.Hist(hist.axis.Regular(bins=50, start=-5, stop=5, name="delta Z[mm]"))
    differenceZ = calculate_differenceZ(reco_vtx_vz)
    # flatten the 
    differenceZ = np.concatenate(differenceZ)
    hist_differenceZ.fill(differenceZ)
    plot_hist(hist_differenceZ)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root-path', help='the input folder path for MC', type=str)
    args = parser.parse_args()
    input_root_path = Path(args.input_root_path)
    performance_differenceZ(input_root_path)
