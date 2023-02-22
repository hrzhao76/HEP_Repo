#!/usr/bin/env bash

workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/
output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/JESJER/

source /global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate ml
python ${workdir}/make_histogram_new.py --write-log \
--do-systs --systs-type JESJER --systs-subtype $1 \
--output-path ${output}/$1 