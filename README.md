## DREAM Olfaction Prediction models and related resources

### The results for and analysis of this challenge appeared in the journal *Science*, entitled: [Predicting human olfactory perception from chemical features of odor molecules](https://dx.doi.org/10.1126/science.aal2014).

### This project is structured as:  

* `data/` (Where the data files go)

     Data files from the challenge, the post-challenge phase, or used for generating figures and tables in the paper.  Essential files for the challenge itself include:

        CID_testset.txt
        CID_leaderboard.txt
        dilution_testset.txt
        dilution_leaderboard.txt
        molecular_descriptors_data.txt
        TrainSet.txt
        leaderboard_set.txt
        LBs1.txt
        LBs2.txt

  ** `derived/` (Data derived from raw data or from other available sources; should be the result of code executed against data stored in `data` or elsewhere online.  

* `predictions/` (Where some prediction files were written during the challenge and post-challenge phase)

* `opc_python` (Python code for loading the data, generating models, and writing prediction files).  

    Currently there are at least three sub-packages here, all written by Rick Gerkin:  

  * `utils` (Generic loading and scoring utilities that should work equally well for all approaches).  
  * `gerkin` (Modules and workflows used by team IKW Allstars, which can optionally be used by other collaborators).  
    * Here are full example workflows for sub-challenge [1](https://github.com/dream-olfaction/olfaction-prediction/blob/master/opc_python/gerkin/challenge1.ipynb) and [2](https://github.com/dream-olfaction/olfaction-prediction/blob/master/opc_python/gerkin/challenge2.ipynb) using both 3a and 3b above.  The code for executing these predictions has since been superseded and should be executed against the appropriate version.  
    * These workflows use Jupyter notebooks running IPython kernels.  Other languages are also supported in Jupyter.  The links above show read-only renderings of the workflows, but they can be interacted with on any machine running an Jupyter notebook server.  
  * `paper` (Notebooks used to produce figures for the journal article).
    * These are numbered according to the final figure numbers and panel letters of the publishing journal article.  These correspond only to figures primarily produced by Rick Gerkin.  Other authors who have not contributed reproduction workflows for the figures they were primarily responsible for generating may be contacted through the corresponding author on the article. Dependencies include numpy, scipy, matplotlib, and pandas.  

  Other python-based files are in sub-directories of the opc_python package, e.g.
  * `john_doe` (Modules and workflows used by John Doe)

Files using other languages by other challenge participants are in subdirectories of this project, e.g.
* `opc_matlab` (A MATLAB package for loading the data, generating models, and writing prediction files).  
* `opc_R` (An R package for loading the data, generating models, and writing prediction files).  

### Scripts used by the organizers to score original challenge submissions are [here](https://github.com/Sage-Bionetworks/OlfactionDREAMChallenge/tree/90adc4695cae6adb0e40222d21e2619b5b776ea0/src/main/resources).
<hr>

## Quick Start:

There is a [Quick Start](https://github.com/dream-olfaction/olfaction-prediction/blob/thin/opc_python/paper/quick-start.ipynb) file that you can use to quickly train the DREAM model, evaluate it, and start making predictions if you have molecules with calculated features.  The most reliable way to run it is using the Docker, with the instructions [here](https://github.com/dream-olfaction/olfaction-prediction/blob/thin/docker/README.md).
