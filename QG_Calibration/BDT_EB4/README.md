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

# Questions   

- [ ] The feature correlation?   
    known: ntrk = 0, c1 = width = -1    
    if we use track associated with jet as feature, we should exclude ntrk = 0.    

- [ ] How to properly implemented the weights? `sqrt(sum(event**2))`?

- [ ] For other tagger, what is the difference between `predict_proba` and `decision_functions` ? -> Check the source code.  

