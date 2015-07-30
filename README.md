# olfaction-prediction
Collaborative Phase of DREAM Olfaction Prediction Challenge

This project is structured as:  

1) `data/` (Where the data files go)

Until release of the paper, we cannot put the training data files in this repository.  So please do not commit them here if you have them.  Instead, obtain the data files through Synapse, and place them locally in the corresponding 'data' directory of this repository.  The files to put here are:

```
CID_testset.txt
CID_leaderboard.txt
molecular_descriptors_data.txt
TrainSet.txt
leaderboard_set.txt
dilution_testset.txt
dilution_leaderboard.txt
LBs1.txt
LBs2.txt
```

You may put more other files in this directory as well, and commit those other files, as long as they are not derivative from the files above or anything else related to the original dataset.  

2) `predictions/` (Where prediction files should be written)

For now these are set to be ignored by the repository, but you should write them in this directory of your local repository so you can add the prediction files of interest when you are ready to have them scored.  

3) `olf_pred` (A python package for loading the data, generating models, and writing prediction files).  

Currently there are two sub-packages here:  

3a) `utils` (Generic loading and scoring utilities that should work equally well for all approaches).  
3b) `gerkin` (Modules and workflows used by Rick Gerkin, which can optionally be used by other collaborators; relies on 3a).  

Here are full example workflows for sub-challenges [1](https://github.com/dream-olfaction/olfaction-prediction/blob/master/olf_pred/gerkin/challenge1.ipynb) and [2](https://github.com/dream-olfaction/olfaction-prediction/blob/master/olf_pred/gerkin/challenge2.ipynb) using both 3a and 3b above.  

Other python-based files should be placed in sub-directories of the olf_pred package.  Files using other languages should be placed in new subdirectories of this project.  
