period=${1}
slice=${2}
workdir=/global/cfs/projectdirs/atlas/hrzhao/qgcal/New_Codes/Validation
output=${workdir}/CheckData${period}${slice}

echo ${workdir}
echo ${output}

mkdir ${output}
find /global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples/Data_lxplus/user.rqian.June1-dijet-data.data${period}_13TeV.period${slice}.physics_Main_minitrees.root/ \
-type f -name "*.root" > ${output}/data${period}${slice}.txt

python ${workdir}/ReadingTree_data.py ${period} ${slice}
python ${workdir}/Make_Histogram_Data.py --path Samples/Data_lxplus/ --period ${period} --slice ${slice}
python ${workdir}/Plot_PtSpectrum_Data_compared.py --period ${period} --slice ${slice}
