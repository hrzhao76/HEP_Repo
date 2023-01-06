# Goal 
This folder is used to develop a new BDT tagger for Q/G Calibration EB4. 

# TODO
- [ ] check subset distribution with the whole dataset 
    - [ ] If doing sampling from a range, does the shape match?  
- [ ] add Forward/Central seperation and check the shape 
- [ ] add a simple extraction and check the extracted (MC Closure)
- [ ] restructure the code, to enable 
    - [ ] The full train chain: read the data, train, check performance: predictions, forward/central, MC Closure etc.. 
    - [ ] Function tuning, current manual setting is to slow. 
- [ ] Check other tagger
    - [ ] GradientBoostingClassifier faster, shape seems ok 
    - [ ] SKlearn MLPClassifier
          - Much faster 
          - Use `pred_proba` as output, shape seperate 
          - However, no sample weights are available (How to properly apply event weight to a tagger? )
    - [ ] PyTorch Fully Connectted 
 

# Tuning
## BDT
possible tuning parameter 
### Base Estimator: Decision Tree
1. max_depth
2. min_samples_split
3. min_samples_leaf
4. max_features

### Ensemble Method 
`learning_rate` and `n_estimators` trade-off

`n_estimators` does change the shape. [config2](./config2) is used to explore the `n_estimators` changes 

### Some Experience  
1. 3 vars : `ntrk` `c1` and `width` forming Forward/Central shape are very robost. 
2. BDT shape starts with two extreme peaks at -1, 1, increasing `learning_rate` or `n_estimators` moves the shape towards 0. 
3. For `GradientBoostingClassifier` in `scikit-learn`, the base estimator is `DecisionTreeRegressor`, the pred output is not discreate 0/1. 
4. If using `predict_proba`, a softmax function is used so the distribution is very seperatable. This is not good for the matrix method.  
# Questions   

- [x] The feature correlation?   
    known: ntrk = 0, c1 = width = -1    
    if we use track associated with jet as feature, we should exclude ntrk = 0.    
    What is the fraction of ntrk = 0 in data?   
    Filter the jets with >=2 trks.   

- [ ] How to properly implemented the weights? `sqrt(sum(event**2))`?

- [x] For other tagger, what is the difference between `predict_proba` and `decision_functions` ? -> Check the source code.    
```python
y_sample_proba = softmax( np.vstack([-y_sample_score, y_sample_score]).T / 2 , axis = 1)
```

- [x] If the fraction of q/g are different in Forward/Central region, is it possible to have smooth curve as a mix of quark/gluon even they are smooth? 
    Look at the MLP prediction, q/g are sperated very well. But in terms of forward/central, no good. Extraction no good. 

# Working Log
## 2023/01/05 - 2023/01/06

Restructure the code. Use the same analysis function here as that for ntrk.   
To do this, use import and convert the prediction to unumpy array.    
Create training_utils, plotting_utils in python, use notebooks to call it and save figures.   
notebooks used to show all plots in one place.      
