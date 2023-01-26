import argparse
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import uproot
from tqdm.auto import tqdm
from itertools import repeat

def get_match_idx(truth_flat_event, reco_flat_event):
    match_idx = []

    truth_trk_table = truth_flat_event[['d0','z0','phi','theta','qp']].to_numpy()
    reco_trk_table = reco_flat_event[['d0','z0','phi','theta','qp']].to_numpy()

    reco_vtxID_list = reco_flat_event['vtxID'].to_numpy()

    for truth_trk in truth_trk_table:
        reco_idx = np.flatnonzero((reco_trk_table == truth_trk).all(1))
        if reco_idx.size != 0:
            match_idx.append(reco_vtxID_list[reco_idx[0]]) # Here we just save the first
        else:
            match_idx.append(-1)

    return match_idx

def root_to_pickle(root_data_path, raw_data_dir):
    if not raw_data_dir.exists():
        raw_data_dir.mkdir(parents=True, exist_ok=True)
    ni = 0
    for f in sorted(root_data_path.glob('*.root')):
        print(f)
        root_dir = uproot.open(f)
        truth_tree = root_dir['Truth_Vertex_PV_Selected']
        reco_tree = root_dir['Reco_Vertex']
        truth_jagged_dict = {}
        reco_jagged_dict = {}
        truth_prefix = 'truth_vtx_fitted_trk_'
        reco_prefix = 'reco_vtx_fitted_trk_'

        for k, v in tqdm(truth_tree.items()):
            if not k.startswith(truth_prefix):
                continue
            truth_jagged_dict[k[len(truth_prefix):]] = v.array()
        
        for k, v in tqdm(reco_tree.items()):
            if not k.startswith(reco_prefix):
                continue
            reco_jagged_dict[k[len(reco_prefix):]] = v.array()
        
        truth_jagged_dict['truth_vtxID'] = truth_jagged_dict.pop('vtxID')

        coords = ['d0', 'z0', 'phi', 'theta', 'qp']
        for n in tqdm(range(len(truth_tree[0].array()))):
            truth_df_dict = {k: truth_jagged_dict[k][n] for k in truth_jagged_dict.keys()}
            reco_df_dict = {l: reco_jagged_dict[l][n] for l in reco_jagged_dict.keys()}
            
            truth_flat_event = pd.DataFrame(truth_df_dict)
            reco_flat_event = pd.DataFrame(reco_df_dict)
            truth_flat_event['truth_semantic_label'] = [0] * len(truth_flat_event)
            truth_flat_event['x0'] = truth_flat_event['d0'] * np.cos(truth_flat_event['phi'] )
            truth_flat_event['y0'] = truth_flat_event['d0'] * np.sin(truth_flat_event['phi'] )

            match_idx = get_match_idx(truth_flat_event, reco_flat_event)
            truth_flat_event['reco_AMVF_vtxID'] = match_idx
            truth_flat_event['reco_semantic_label'] = [0] * len(truth_flat_event)
            
            idx_not_found = truth_flat_event['reco_AMVF_vtxID'] == -1
            truth_flat_event.loc[idx_not_found,'reco_semantic_label'] = [0]*len(truth_flat_event['reco_semantic_label'].loc[idx_not_found])
            
            truth_flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
        ni += n + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', help='root folder path', type=str)
    parser.add_argument('--output-path', help='the output folder path', type=str)

    args = parser.parse_args()

    root_data_path = Path(args.sample_path)
    if args.output_path is None: 
        output_path = root_data_path / "flatten_events"
    else:
        output_path = args.output_path

    root_to_pickle(
        root_data_path=root_data_path,
        raw_data_dir=output_path,
    )
