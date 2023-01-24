# Intro slide 
[SPVCNN Workflow](https://docs.google.com/presentation/d/1qR9GRVWcSvsXeMRz6mmUiR5hK7s0aumYdTavAWu5EDw/edit#slide=id.p)

# Code Commands 
## Training SPVCNN

``` python
python train.py configs/vertexing_panoptic/spvcnn/objectcondensation_dzp_pu10.yaml num_epochs=10 --local dataset.train_frac=0.001 dataset.test_frac=0.998 
```

## Evaluate performance
`scripts/write_vertexing.py` saves the SPVCNN predicted cluster ID to pkl files as the input to the `scripts/pkl2root.py`. It also plots image comparisons between truth, AMVF and SPVCNN cluster by event image.  

```
PYTHONPATH=. python scripts/write_vertexing.py --run-dir ./tmp/ --checkpoint-path ./tmp/checkpoints/step-50.pt
```
### Evalueate PQ score 
inspect PU10 dataset
``` python 
python evaluate/evaluate_pq_score_vertexing.py --run-dir /global/cscratch1/sd/hrzhao/calo_cluster/data/vertex/raw_original_new
#### output 
#### ious: [1.0]
#### mious: 1.0
#### sq: [0.90117917]
#### msq: 0.9011791694854425
#### rq: [0.88927486]
#### mrq: 0.8892748588797221
#### pq: [0.80139598]
#### mpq: 0.8013959787695121
#### 0.8013959787695121
```


## Convert prediction to ROOT TTree 
`scripts/pkl2root.py` converts the predicted SPVCNN cluster ID to to TTree format, to be called by ACTS AMVF Fitter algorithm. This script generates a root file in the given folder.  

```
python scripts/pkl2root.py tmp/preds/tbeta=0.75_td=0.4_pkls
```


## ACTS framework
The easiest way to build acts on nersc may be using `shifter`. Open a new terminal, otherwise some environment variables are conflicted with cgpu. 


``` bash
git clone https://github.com/hrzhao76/acts.git 
cd acts 
git checkout dev_stdalone_fitter 

shifter --image=tomohiroyamazaki/acts-ubuntu2004:v29 /bin/bash 

cmake ../  -DACTS_BUILD_EXAMPLES_PYTHIA8=ON


./bin/SPVCNN_AdaptiveMultiVertexFitter /global/cfs/projectdirs/atlas/hrzhao/spvnas-dev/tmp/preds/tbeta=0.75_td=0.4_pkls/SPVCNN_outputs.root 
```

## Evaluate Physics performance 

One can use these two notebooks to compare SPVCNN and AMVF on HS vertex reconstruction efficiency and identification efficiency, and primary vertex classification.   
[Compare_eff_sel.ipynb](https://github.com/hrzhao76/HEP_Repo/blob/master/SPVCNN/performance/Compare_eff_sel.ipynb)   

[Compare_pv_classification.ipynb](https://github.com/hrzhao76/HEP_Repo/blob/master/SPVCNN/performance/Compare_pv_classification.ipynb)

