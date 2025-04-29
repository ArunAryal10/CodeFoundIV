# Foundation IV project repository - Permutation testing using MaxT 

This project aims to perform permutation testing using MaxT approach (https://doi.org/10.1002/hbm.1058) to control for family-wise errors(FWE).

**Hypothesis**:
The structure of functional connectivity (FC) and task activation enables above-chance decoding of somatomotor network (SMN) 
activity patterns during task performance. When FC and activation patterns are randomly shuffled, this information structure is disrupted, and 
decoding performance drops to chance.

**Dataset**: 
The dataset consists of 36 subject's task activation across 8 runs while performing NEXT behavioral task. 
<img width="783" alt="image" src="https://github.com/user-attachments/assets/a9e65554-3227-4a18-b7ec-7d852fea0936" />


**Method**:
1. Task activations were estimated using regularized ridge regression (https://github.com/alexhuth/ridge). 
2. Parcel-level functional connectivity was estimated using Graphical Lasso regression **(Peterson, 2023)**.
3. A generative multi-step activity flow model was used to create trial-wise predicted activations using FPN as 
task instruction encoding network and VIS1/VIS2 as GO probe encoding network.
<img width="535" alt="image" src="https://github.com/user-attachments/assets/bd9c6f7f-b97d-4c37-a033-419cdfd7f4d3" />

5. Two way networkwise button press decoding was performed on actual betas to estimate the ground truth.
6. Two way networkwise button press decoding was performed on predicted betas and was tested for significance. 
7. FC was shuffled (1000 permutations) and max t-stat null distribution of accuracy was estimated. 
8. Trial-wise activation was shuffled (1000 permutations) and max t-stat null distribution of accuracy was estimated. 

**Result**: 
Both permutations showed non-shuffled decoding on SMN to be significantly different compared to null distribution (p < 0.05),
hence confirming the hypothesis.



