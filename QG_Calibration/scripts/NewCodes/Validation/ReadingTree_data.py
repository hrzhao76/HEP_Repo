#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import uproot3 as uproot
import numpy as np
import sys


bins = [300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]  #this bin range is for only dijet event
#bins = [0, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]  #this bin range is for gammajet+dijet event
HistMap = {}
JetList = []
	   
finput = open(f"/global/cfs/projectdirs/atlas/hrzhao/qgcal/New_Codes/Validation/CheckData{sys.argv[1]}{sys.argv[2]}/data{sys.argv[1]}{sys.argv[2]}.txt")
inputs = finput.read().splitlines()


usefiles = []


for file in inputs:
    try:
        tr = uproot.open(file)["nominal"]
    except:
        print(str(file)+" has no events pass cut!")
    else:
        if( "data"+str(sys.argv[1]) in file):
            print(file)
            usefiles.append(file)
            

###### define functions
def GetHistBin(histname):
	if 'pt' in histname:
		return 60,0,2000
	elif 'eta' in histname:
		return 50,-2.5,2.5
	elif 'ntrk' in histname:
		return 60,0,60
	elif 'bdt' in histname:
		return 60,-0.8,0.7
	elif 'width' in histname:
		return 60,0.,0.4
	elif 'c1' in histname:
		return 60,0.,0.4

def FillTH1F(histname, var, w):
    if histname in HistMap:
        HistMap[histname][0].append(var)
        HistMap[histname][1].append(w) 
    else:
        HistMap[histname] = [[],[]] #The first list is for the data, the second for the weights
        HistMap[histname] = [[],[]]
        HistMap[histname][0].append(var)
        HistMap[histname][1].append(w)

def FillHisto(prefix, jetlist, w):
	FillTH1F(prefix+"_ntrk", jetlist[0], w)
	FillTH1F(prefix+"_bdt", jetlist[1], w)
	FillTH1F(prefix+"_width", jetlist[2], w)
	FillTH1F(prefix+"_c1", jetlist[3], w)
	FillTH1F(prefix+"_pt", jetlist[4], w)
	FillTH1F(prefix+"_eta", jetlist[5], w)


def GetJetType(label):
	if label == -9999:
		return "Data"
	elif label == 21:
		return "Gluon"
	elif label > 0 and label < 4:
		return "Quark"
	elif label == 4:
		return "C_Quark"
	elif label == 5:
		return "B_Quark"
	else:
		return "Other"

def FindBinIndex(jet_pt,ptbin):
	for j in range(len(ptbin)-1):
		if jet_pt >= ptbin[j] and jet_pt < ptbin[j+1]:
			return ptbin[j]

	print("error: jet pT outside the bin range")
	return -1

#Unfourtunately I can't fully utilize the use of arrays because each jet must be matched with the corresponding histogram.
#for i in range():


def ReadTree(df,file):    
    for i in range(0,len(df[b"jet_fire"])):
        if(df[b"jet_fire"][i] == 1 and len(df[b"jet_pt"][i]) > 1 and df[b"jet_pt"][i][0]/1000 > 500 and df[b"jet_pt"][i][0]/1000 < 2000 and abs(df[b"jet_eta"][i][0]) < 2.1 and abs(df[b"jet_eta"][i][1]) < 2.1 and df[b"jet_pt"][i][0]/df[b"jet_pt"][i][1] < 1.5):
            
            pTbin1 = FindBinIndex(df[b"jet_pt"][i][0]/1000, bins)
            pTbin2 = FindBinIndex(df[b"jet_pt"][i][1]/1000, bins)

            label1 = GetJetType(df[b"jet_PartonTruthLabelID"][i][0])
            label2 = GetJetType(df[b"jet_PartonTruthLabelID"][i][1])

            eta1 = "Central"
            eta2 = "Forward"
            if abs(df[b"jet_eta"][i][0]) > abs(df[b"jet_eta"][i][1]):
                eta1 = "Forward"
                eta2 = "Central"
            
            JetList = [[df[b"jet_nTracks"][i][0], df[b"jet_trackBDT"][i][0], df[b"jet_trackWidth"][i][0], df[b"jet_trackC1"][i][0], df[b"jet_pt"][i][0]/1000, df[b"jet_eta"][i][0]],[df[b"jet_nTracks"][i][1], df[b"jet_trackBDT"][i][1], df[b"jet_trackWidth"][i][1], df[b"jet_trackC1"][i][1], df[b"jet_pt"][i][1]/1000, df[b"jet_eta"][i][1]]]
            	
            total_weight = 1

            if sys.argv[1] == "18":
                total_weight = 58.45/39.91
            	
            FillHisto(str(pTbin1)+"_LeadingJet_"+eta1+"_"+label1, JetList[0], total_weight)
            FillHisto(str(pTbin2)+"_SubJet_"+eta2+"_"+label2, JetList[1], total_weight)


######## read and excute TTree from root file 
#finput = TFile.Open("/eos/user/e/esaraiva/AQT_dijet_sherpa_bdt/dijet_sherpa_bdt_d.root")

#print(usefiles)
for file in usefiles:
    print(file)
    tr = uproot.open(file)["nominal"]
    data = tr.arrays(["jet_fire","jet_pt","jet_eta","jet_nTracks","jet_trackWidth","jet_trackC1","jet_trackBDT","jet_PartonTruthLabelID"])
    ReadTree(data,file)

foutput = uproot.recreate(f"/global/cfs/projectdirs/atlas/hrzhao/qgcal/New_Codes/Validation/CheckData{sys.argv[1]}{sys.argv[2]}/dijet_data_OLD_{sys.argv[1]}{sys.argv[2]}.root")

#Create the actual histograms now that the data is in separate lists
#uproot lets you use numpy histograms and write them to root files.
for hist in HistMap.keys():
    #print(HistMap)
    nbin,binmin,binmax = GetHistBin(hist)
    histogram = np.histogram(a = HistMap[hist][0], weights = HistMap[hist][1], bins = nbin, range = (binmin,binmax))
    #print(histogram)
    foutput[hist] = histogram
    
    weight = np.array(HistMap[hist][1])
    binning = np.linspace(binmin,binmax,nbin)
    sum_w2 = np.zeros([nbin], dtype=np.float32)
    digits = np.digitize(HistMap[hist][0],binning)
    for i in range(nbin):
        weights_in_current_bin = weight[np.where(digits == i)[0]]
        sum_w2[i] = np.sum(np.power(weights_in_current_bin, 2))
    #print(sum_w2)
    histogram_err = np.histogram(a = binning, weights = sum_w2, bins = nbin, range = (binmin,binmax))
    foutput[hist+"_err"] = histogram_err
