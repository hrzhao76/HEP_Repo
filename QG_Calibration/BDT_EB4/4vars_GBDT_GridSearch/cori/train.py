# %%
import os, sys 
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
sys.path.append('/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/')
from BDT_train import *

# %%
cv_result_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/4vars_GBDT_GridSearch/cv_results_flat_pt.pkl'
with open(cv_result_path, 'rb') as f:
    cv_result = pickle.load(f)

# %%
np.max(cv_result['mean_test_score'])

# %%
cv_result = pd.DataFrame(cv_result)

# %%
print(cv_result['params'][cv_result['rank_test_score']==1].values[0])

# %%
output_path = './'
sample_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/samples/sample_allpt_all_jets.pkl'
training_vars = ['jet_pt', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1']
training_weights = 'flatpt_weight'
n_estimators = cv_result['params'][cv_result['rank_test_score']==1].values[0]['n_estimators']
learning_rate = cv_result['params'][cv_result['rank_test_score']==1].values[0]['learning_rate']
max_depth = cv_result['params'][cv_result['rank_test_score']==1].values[0]['max_depth']


# %%
main(output_path=output_path, sample_path=sample_path, 
    training_vars=training_vars, training_weights=training_weights,
    n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)


# %%




