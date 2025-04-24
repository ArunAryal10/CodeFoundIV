# Foundation IV projects repository

This project aims to perform permutation testing using MaxT approach **(Nichols & Holmes, 2001)** to control for family-wise errors(FWE).

The dataset consists of 36 subject's task activation across 8 runs while performing NEXT behavioral task. 
Task activations were estimated using regularized ridge regression (https://github.com/alexhuth/ridge). 
Parcel-level functional connectivity was estimated using Graphical Lasso regression **(Peterson, 2023)**.

A generative multi-step activity flow model was used to create trial-wise predicted activations using FPN as 
task instruction encoding network and VIS1/VIS2 as GO probe encoding network.

Two way button press decoding was performed on actual betas to estimate the ground truth.

Two way networkwise button press decoding was performed on predicted betas and was tested for significance. 

FC was shuffled (1000 permutations) and max t-stat null distribution of accuracy was estimated. 

Trial-wise activation was shuffled (1000 permutations) and max t-stat null distribution of accuracy was estimated. 

Both permutations showed non-shuffled decoding on SMN to be significantly different compared to null distribution (p < 0.05).





