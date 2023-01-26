from root2pkl import *
from concurrent.futures import ProcessPoolExecutor
import functools 


# def all_in_one():
#     #### Change some parameter here
#     root2pkl = functools.partial(root2pkl, output_path='./processed_pythia', verbosity=2, write_log=False)
#     root_files = []
#     for period in ["A", "D", "E"]:
#         pythia_path = f'/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/pythia{period}/'
#         pythia_path = Path(pythia_path)

#         root_files.append(sorted(pythia_path.rglob("*JZ?WithSW_minitrees.root/*.root")))

#     flat_list = list(np.concatenate(root_files).flat)

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         results = list(executor.map(root2pkl, flat_list))

def split_ADE():
    for period in ["A", "D", "E"]:
        pythia_path = f'/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/pythia{period}/'
        pythia_path = Path(pythia_path)

        root_files = sorted(pythia_path.rglob("*JZ?WithSW_minitrees.root/*.root"))

        pkl_output_path = pythia_path.parent / (pythia_path.stem + "_pkl")

        root2pkl_mod = functools.partial(root2pkl, output_path=pkl_output_path, verbosity=2, write_log=False)
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(root2pkl_mod, root_files)
    
def split_Data():
    for period in ["1516", "17", "18"]:
        data_path = f"/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_New/data/data{period}"
        data_path = Path(data_path)

        root_files = sorted(data_path.rglob("*data*_13TeV.period?.physics_Main_minitrees.root/*.root"))

        pkl_output_path = data_path.parent / (data_path.stem + "_pkl")

        root2pkl_mod = functools.partial(root2pkl, output_path=pkl_output_path, verbosity=2, write_log=False)
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(root2pkl_mod, root_files)

        

if __name__ == "__main__":
    # all_in_one()
    split_ADE()
    split_Data()