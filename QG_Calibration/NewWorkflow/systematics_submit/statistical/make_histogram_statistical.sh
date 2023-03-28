#!/usr/bin/env bash
syst_identifier=statistical

input_file=${1}
workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/${syst_identifier}/
output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/${syst_identifier}/

source /global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate ml

python ${workdir}/bootstrap.py  \
--mode parallel \
--output-path ${output} \
--input-file ${input_file}
