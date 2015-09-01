feature_selection.py selects the best features with Lasso 

It loads the targets_for_feature_selection.csv file. This csv should contain the preprocessed target data (whichever we want to calculate with) in the form of molecule ID x descriptor data.

creation_of_targets_for_feature_selection_train_data_only.py and creation_of_targets_for_feature_selection_with_LB_data.py creates this file by preprocessing and averaging the two intensities for everything except intensity. 

feature_selection.py also needs the molecular_descriptors_data.txt, CIDs.txt (train, leaderboard and test CIDs) and all_smiles.csv (molecule smiles preproduced) files. Find the latter two in this folder.

creation_of_targets_for_feature_selection scripts need TrainSet.txt (and LBs1.txt, leaderboard_set_Low_Intensity.txt if calculating for final prediction).


