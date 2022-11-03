# Open multiple terminals to run this script 
# e.g.
# ./make_Histogram_Data_single_period.sh 17
python -u New_Codes/ReadingTree/make_Histogram_Data.py \
--path Samples/Data_lxplus --period ${1} \
--output-path ./Processed_Samples_Data/data${1} \
--reweight-file-path ./Processed_Samples \
2>&1 | tee ./Processed_Samples_Data/data${1}/log.process.data${1}.txt
