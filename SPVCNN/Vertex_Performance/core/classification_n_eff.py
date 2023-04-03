import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ROOT
import uproot 
import awkward as ak
from pathlib import Path
import os
import re
from tqdm import tqdm


from utils import check_outputpath
from utils import vxMatchWeight, cutMinTruthRecoRadialDiff, VertexMatchType, HardScatterType

def Get_local_PU_density(truth_vtx_vx, truth_vtx_vy, truth_vtx_vz, i_event, xyz_dist_window = 2.0):
    # Calculate the PU density around the truth HS vertex 
    residual_truth_vtx_vx = truth_vtx_vx[i_event] - truth_vtx_vx[i_event][0]
    residual_truth_vtx_vy = truth_vtx_vy[i_event] - truth_vtx_vy[i_event][0]
    residual_truth_vtx_vz = truth_vtx_vz[i_event] - truth_vtx_vz[i_event][0]
    
    dist_to_truth_HS = residual_truth_vtx_vx**2 + residual_truth_vtx_vy**2 + residual_truth_vtx_vz**2 
    n_local_truth = len(np.where(dist_to_truth_HS< xyz_dist_window**2)[0])
    return (n_local_truth - 1)/(2 * xyz_dist_window)

def Get_reco_info(reco_tree_arrays, type="truth_refitting"):
    reco_type = "reco_"

    if type == "truth_refitting":
        reco_vtx_vx = reco_tree_arrays["fitted_avg_pos_vtx_vx"]
        reco_vtx_vy = reco_tree_arrays['fitted_avg_pos_vtx_vy']
        reco_vtx_vz = reco_tree_arrays['fitted_avg_pos_vtx_vz']

        reco_trk_z0 = reco_tree_arrays['reco_vtx_fitted_trk_z0']
        reco_trk_qp = reco_tree_arrays["reco_vtx_fitted_trk_qp"]
        reco_trk_theta = reco_tree_arrays["reco_vtx_fitted_trk_theta"]
        reco_trk_trackWeight = reco_tree_arrays["reco_vtx_fitted_trk_trackWeight"]
        reco_trk_vtxID = reco_tree_arrays["reco_vtx_fitted_trk_vtxID"]
    
    if type == "amvf":
        reco_vtx_vx = reco_tree_arrays["reco_vtx_vx"]
        reco_vtx_vy = reco_tree_arrays['reco_vtx_vy']
        reco_vtx_vz = reco_tree_arrays['reco_vtx_vz']

        reco_trk_z0 = reco_tree_arrays['reco_vtx_fitted_trk_z0']
        reco_trk_qp = reco_tree_arrays["reco_vtx_fitted_trk_qp"]
        reco_trk_theta = reco_tree_arrays["reco_vtx_fitted_trk_theta"]
        reco_trk_trackWeight = reco_tree_arrays["reco_vtx_fitted_trk_trackWeight"]
        reco_trk_vtxID = reco_tree_arrays["reco_vtx_fitted_trk_vtxID"]


    return reco_vtx_vx, reco_vtx_vy, reco_vtx_vz, reco_trk_z0, reco_trk_qp, reco_trk_theta, reco_trk_trackWeight, reco_trk_vtxID

def classifyHardScatter(RecoVertexMatchInfo:np.ndarray, vtx_types:dict) -> HardScatterType:

    # count how many reco vtx the truth HS contributes to 
    n_contribution_from_truth_HS = np.count_nonzero(RecoVertexMatchInfo[:,0,0])

    if (n_contribution_from_truth_HS==0):
        return HardScatterType.NONE
    elif(n_contribution_from_truth_HS==1):
        # find the only one reco idx that truth HS contributes to 
        reco_vtx_idx = np.flatnonzero(RecoVertexMatchInfo[:,0,0]!=0)[0]
        # check if the 
        # FIXME athena uses the weights but here I use the sum of pt2 
        is_largest_contribution = reco_vtx_idx == np.argmax(RecoVertexMatchInfo[reco_vtx_idx,:,2])
        reco_vtx_type = vtx_types[reco_vtx_idx]
        if is_largest_contribution and reco_vtx_type == VertexMatchType.MATCHED:
            return HardScatterType.CLEAN
        elif is_largest_contribution and reco_vtx_type == VertexMatchType.MERGED:
            return HardScatterType.LOWPU
        else:
            return HardScatterType.HIGHPU
    else: 
        # multiple reco vertices have tracks from hard-scatter
        # count how many have hard-scatter tracks as largest contribution
        reco_vtxs_idx = np.flatnonzero(RecoVertexMatchInfo[:,0,0]!=0)
        largest_contributution_idxs = np.argmax(RecoVertexMatchInfo[reco_vtxs_idx, :, 2], axis=1)
        n_largest_contribution_from_truth_HS = np.count_nonzero(largest_contributution_idxs==0)
        if n_largest_contribution_from_truth_HS == 0:
            return HardScatterType.HIGHPU
        elif n_largest_contribution_from_truth_HS == 1: 
            # Only one reco vtx has the largest contribution
            # identify this reco vtx 
            reco_vtx_idx = reco_vtxs_idx[np.where(largest_contributution_idxs == 0)[0][0]]
            # take its vtx type 
            reco_vtx_type = vtx_types[reco_vtx_idx]
            # choose the event type 
            if reco_vtx_type == VertexMatchType.MATCHED:
                return HardScatterType.CLEAN
            elif reco_vtx_type == VertexMatchType.MERGED:
                return HardScatterType.LOWPU
            else:
                return HardScatterType.HIGHPU
        else:
            return HardScatterType.HSSPLIT
    
def do_matching(truth_tree_arrays, reco_tree_arrays):
    hs_reco_eff = ROOT.TEfficiency("hs_reco_eff", "HS Reconstruction Efficiency; Local PU density; eff", 12, 0, 6)
    hs_sel_eff = ROOT.TEfficiency("hs_sel_eff", "HS Selection and Reconstruction Efficiency; Local PU density; eff", 12, 0, 6)

    test_event_idx = reco_tree_arrays.event_id[0:1000]
    PV_Classification = np.zeros((len(VertexMatchType._member_names_[:-1])),int)
    HS_Classification = np.zeros((len(HardScatterType._member_names_[:-1])),int)
    total_n_reco_vtx = 0
    n_recoed_hs_vtx = 0
    n_recoed_seled_hs_vtx = 0

    truth_vtx_vx = truth_tree_arrays['truth_vtx_vx']
    truth_vtx_vy = truth_tree_arrays['truth_vtx_vy']
    truth_vtx_vz = truth_tree_arrays['truth_vtx_vz']
    truth_trk_z0 = truth_tree_arrays['truth_vtx_fitted_trk_z0']
    truth_trk_vtxID = truth_tree_arrays['truth_vtx_fitted_trk_vtxID']

    reco_vtx_vx, reco_vtx_vy, reco_vtx_vz, reco_trk_z0, reco_trk_qp, reco_trk_theta, reco_trk_trackWeight, reco_trk_vtxID = Get_reco_info(reco_tree_arrays, type="amvf")
    return_RecoVertexMatchInfo = []
    for i, event_id in enumerate(tqdm(test_event_idx)):            
        trk_reco_vtx_ID = ak.to_numpy(reco_trk_vtxID[i])
        trk_truth_vtx_ID = ak.to_numpy(truth_trk_vtxID[i])
        trk_pt_sq = ((1./reco_trk_qp[i])*np.sin(reco_trk_theta[i]))**2

        n_truth_vtx = len(truth_vtx_vz[i])
        n_reco_vtx = len(reco_vtx_vz[i])
        total_n_reco_vtx +=  n_reco_vtx
        Raw_RecoVertexMatchInfo = np.zeros((n_reco_vtx, n_truth_vtx, 3), dtype=float)
        for reco_vtx_id in range(n_reco_vtx):
            for truth_vtx_id in range(n_truth_vtx):
                trk_at_reco_id = np.where(trk_reco_vtx_ID == reco_vtx_id)[0]
                trk_at_truth_id = np.where(trk_truth_vtx_ID == truth_vtx_id)[0]
                intersect1, reco_common_idx1, truth_common_idx2 = np.intersect1d(
                    reco_trk_z0[i][trk_at_reco_id], 
                    truth_trk_z0[i][trk_at_truth_id], 
                    return_indices=True)

                Raw_RecoVertexMatchInfo[reco_vtx_id, truth_vtx_id, 0] = len(intersect1)
                Raw_RecoVertexMatchInfo[reco_vtx_id, truth_vtx_id, 1] = ak.sum(reco_trk_trackWeight[i][reco_common_idx1])
                Raw_RecoVertexMatchInfo[reco_vtx_id, truth_vtx_id, 2] = ak.sum(trk_pt_sq[reco_common_idx1])

        RecoVertexMatchInfo = np.copy(Raw_RecoVertexMatchInfo)
        RecoVertexMatchInfo[:,:,1] = Raw_RecoVertexMatchInfo[:,:,1] / Raw_RecoVertexMatchInfo[:,:,1].sum(axis = 0)
        RecoVertexMatchInfo[np.isnan(RecoVertexMatchInfo)[:,:,1]] = 0

        return_RecoVertexMatchInfo.append(RecoVertexMatchInfo)
        vtx_types = {}
        assigned_type = np.array([-1]*(n_reco_vtx))

        for reco_vtx_id in range(n_reco_vtx):      
            max_weight_idx = np.argmax(RecoVertexMatchInfo[:,:,1][reco_vtx_id])
            if RecoVertexMatchInfo[:,:,1][reco_vtx_id][max_weight_idx] > vxMatchWeight and RecoVertexMatchInfo[:,:,2][reco_vtx_id][max_weight_idx] == Raw_RecoVertexMatchInfo[reco_vtx_id,:,2].max():
                assigned_type[reco_vtx_id] = 0 # labelled as matched/clean 
                vtx_types[reco_vtx_id] = VertexMatchType.MATCHED
            elif RecoVertexMatchInfo[:,:,1][reco_vtx_id][max_weight_idx] < vxMatchWeight and RecoVertexMatchInfo[:,:,2][reco_vtx_id][max_weight_idx] == Raw_RecoVertexMatchInfo[reco_vtx_id,:,2].max():
                assigned_type[reco_vtx_id] = 1 # labelled as merged
                vtx_types[reco_vtx_id] = VertexMatchType.MERGED
            else: 
                assigned_type[reco_vtx_id] = 2 # labelled as spilt 
                vtx_types[reco_vtx_id] = VertexMatchType.SPLIT

        HS_type = classifyHardScatter(RecoVertexMatchInfo, vtx_types)
        HS_Classification[HS_type.value] += 1
        stat = np.bincount(assigned_type)
        for PV_type in range(len(stat)):
            PV_Classification[PV_type] += stat[PV_type]

        ### HS reco eff
        ind_best_reco_HS_nTrk = np.argmax(RecoVertexMatchInfo[:, 0, 0], axis = 0)
        ind_best_reco_HS_sumpt2 = np.argmax(RecoVertexMatchInfo[:,:,2].sum(axis=1))

        residual = np.array([reco_vtx_vx[i][ind_best_reco_HS_nTrk] - truth_vtx_vx[i][0],  
                             reco_vtx_vy[i][ind_best_reco_HS_nTrk] - truth_vtx_vy[i][0], 
                             reco_vtx_vz[i][ind_best_reco_HS_nTrk] - truth_vtx_vz[i][0]])

        ### Get PU density 
        local_PU_density = Get_local_PU_density(truth_vtx_vx=truth_vtx_vx, truth_vtx_vy=truth_vtx_vy, truth_vtx_vz=truth_vtx_vz, i_event = i)
        
        trhth_HS_vtx_recoed = False
        trhth_HS_vtx_recoed_seled = False
        if np.square(residual).sum() <= cutMinTruthRecoRadialDiff ** 2:
            n_recoed_hs_vtx += 1
            trhth_HS_vtx_recoed = True            
            if ind_best_reco_HS_nTrk == ind_best_reco_HS_sumpt2:
                n_recoed_seled_hs_vtx += 1 
                trhth_HS_vtx_recoed_seled = True

        hs_reco_eff.Fill(trhth_HS_vtx_recoed, local_PU_density)
        hs_sel_eff.Fill(trhth_HS_vtx_recoed and trhth_HS_vtx_recoed_seled, local_PU_density)



    assert np.sum(PV_Classification) == total_n_reco_vtx
    return PV_Classification, HS_Classification, n_recoed_hs_vtx, n_recoed_seled_hs_vtx, hs_reco_eff, hs_sel_eff, return_RecoVertexMatchInfo

def plot_pv_hs_classification(enum_Classification, plot_type, 
                              output_path=None, output_name=None, pu_number=None):
    if plot_type == "pv":
        enum_type = VertexMatchType
        identifier_title = "Primary vertex"

    elif plot_type == "hs":
        enum_type = HardScatterType
        identifier_title = "HS event"

    n_types = VertexMatchType.NTYPES.value
    bins_edges = np.arange(0, n_types + 1)

    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    x_labels = enum_type._member_names_[:-1]

    fig, ax = plt.subplots()
    ax.stairs(enum_Classification)
    ax.set_xticks(bin_centers, x_labels)
    ax.set_title(f"{identifier_title} classification on AMVF")
    ax.set_xlabel(f"{identifier_title} type")
    ax.set_ylabel("Number")
    fig.savefig(output_path / output_name)

def plot_eff(eff, plot_type, output_path=None, output_name=None, pu_number=None):
    canvas_eff = ROOT.TCanvas()
    legend_eff = ROOT.TLegend(0.1,0.2,0.4,0.4)

    eff.SetLineColor(2)
    eff.Draw()
    legend_eff.AddEntry(eff, "AMVF")
    legend_eff.Draw("same")
    canvas_eff.Draw()

    canvas_eff.Print((output_path /f"{output_name}.png").as_posix())

def performance_classification_n_eff(
        input_root_path:Path,
        output_path:Path,
        reco_tree_name='Reco_Vertex',
        truth_tree_name='Truth_Vertex_PV_Selected',
        ):
    output_path = output_path / "classification_n_eff"
    check_outputpath(output_path)

    ### Reading the information from TTree
    root_file =uproot.open(input_root_path)
    file_name = input_root_path.stem # for example, vertexperformance_AMVF_pu100.root
    pu_search_pattern = r'pu(.+)$'
    pu_number = re.search(pu_search_pattern, file_name).group((1))

    truth_tree_arrays = root_file[truth_tree_name].arrays()
    amvf_tree_arrays = root_file[reco_tree_name].arrays()

    PV_Classisfication, HS_Classification, n_recoed_hs_vtx, n_recoed_seled_hs_vtx, hs_reco_eff, hs_sel_eff, RecoVertexMatchInfo = do_matching(truth_tree_arrays, reco_tree_arrays=amvf_tree_arrays)
    
    plot_pv_hs_classification(PV_Classisfication, plot_type="pv", output_path=output_path, output_name="pv_classification")
    plot_pv_hs_classification(HS_Classification, plot_type="hs", output_path=output_path, output_name="hs_classification")

    plot_eff(hs_reco_eff, plot_type="hs_reco_eff", output_path=output_path, output_name="hs_reco_eff")
    plot_eff(hs_sel_eff, plot_type="hs_sel_eff", output_path=output_path, output_name="hs_sel_eff")

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




