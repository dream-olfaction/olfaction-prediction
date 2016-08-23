## DREAM Olfaction Prediction models and related resources

### This project is structured as:  

* `data/` (Where the data files go)

    Until release of the paper, we cannot put the training data files in this repository.  So please do not commit them here if you have them.  Instead, obtain the data files through Synapse, and place them locally in the corresponding 'data' directory of this repository.  The files already here with permission are:

        CID_testset.txt
        CID_leaderboard.txt
        dilution_testset.txt
        dilution_leaderboard.txt
        molecular_descriptors_data.txt

    The files *NOT* to commit here contain perceptual descriptor data from the challenge.  They should be download from Synapse and stored only locally.  These include:

        TrainSet.txt
        leaderboard_set.txt
        LBs1.txt
        LBs2.txt

    You may put more other files in this directory as well, and commit those other files, as long as they are not derivative from the files above or anything else related to the original dataset.  

  ** `derived/` (Data derived from raw data or from other available sources; should be the result of code executed against data stored in `data` or elsewhere online.  

* `predictions/` (Where prediction files should be written)

    For now these are set to be ignored by the repository, but you should write them in this directory of your local repository so you can add the prediction files of interest when you are ready to have them scored.  

* `opc_python` (A python package for loading the data, generating models, and writing prediction files).  

    Currently there are two sub-packages here:  

  * `utils` (Generic loading and scoring utilities that should work equally well for all approaches).  
  * `gerkin` (Modules and workflows used by Rick Gerkin, which can optionally be used by other collaborators; relies on 3a).  
  * `paper` (Notebooks used to produce figures for the journal article).

    Here are full example workflows for sub-challenge [1](https://github.com/dream-olfaction/olfaction-prediction/blob/master/opc_python/gerkin/challenge1.ipynb) and [2](https://github.com/dream-olfaction/olfaction-prediction/blob/master/opc_python/gerkin/challenge2.ipynb) using both 3a and 3b above.  These workflows use Jupyter notebooks running IPython kernels.  Other languages are also supported in Jupyter.  The links above show read-only renderings of the workflows, but they can be interacted with on any machine running an Jupyter notebook server.  

  Other python-based files should be placed in sub-directories of the opc_python package, e.g.
  * `john_doe` (Modules and workflows used by John Doe)

Files using other languages should be placed in new subdirectories of this project, e.g.
* `opc_matlab` (A MATLAB package for loading the data, generating models, and writing prediction files).  
* `opc_r` (An R package for loading the data, generating models, and writing prediction files).  

### Scripts used by the organizers to score original challenge submissions are [here](https://github.com/Sage-Bionetworks/OlfactionDREAMChallenge/tree/90adc4695cae6adb0e40222d21e2619b5b776ea0/src/main/resources).
