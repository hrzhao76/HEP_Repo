python New_Codes/PtSpectrum/pt_spectrum_period.py \
--mcpath ./Processed_Samples/dijet_pythia_mc16A.root \
--datapath ./Processed_Samples_Data/data1516/dijet_data_1516.root  \
--period A --output-path New_Codes/PtSpectrum/pt_spectrum_A/


python New_Codes/PtSpectrum/pt_spectrum_period.py \
--mcpath ./Processed_Samples/dijet_pythia_mc16D.root \
--datapath ./Processed_Samples_Data/data17/dijet_data_17.root  \
--period D --output-path New_Codes/PtSpectrum/pt_spectrum_D/


python New_Codes/PtSpectrum/pt_spectrum_period.py \
--mcpath ./Processed_Samples/dijet_pythia_mc16E.root \
--datapath ./Processed_Samples_Data/data18/dijet_data_18.root  \
--period E --output-path New_Codes/PtSpectrum/pt_spectrum_E/

python New_Codes/PtSpectrum/pt_spectrum_period.py \
--mcpath ./Processed_Samples/dijet_pythia_mc16ADE.root \
--datapath ./Processed_Samples_Data/dijet_data_all.root  \
--period ADE --output-path New_Codes/PtSpectrum/pt_spectrum_ADE/