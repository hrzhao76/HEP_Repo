from root2pkl import *


if __name__ == "__main__":
    pythiaA_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/pythiaA/'
    pythiaA_path = Path(pythiaA_path)

    root_files = sorted(pythiaA_path.rglob("*JZ3WithSW_minitrees.root/*.root"))

    for file in root_files:
        root2pkl(file)