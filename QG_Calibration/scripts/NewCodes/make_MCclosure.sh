for reweight_factor in none quark gluon
do
    echo ${reweight_factor}
    python ./New_Codes/Extraction/MCclosure.py \
    --path ./Processed_Samples/dijet_pythia_mc16ADE.root --period ADE \
    --reweighting ${reweight_factor} --output-path ./Processed_Samples
done