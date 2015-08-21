similarity_feature_calculation script calculates the similarities between the compounds using rdkit library. Test compound are compared with test compound and a set of other odorants. Odorant molecule list was collected from www.odour.org.uk and www.immuneweb.org/articles/fragrancelist.html.

Morgan fingerprints have very high dimensionality themselves, so only the similarities were used. CIDs.txt contains the odorant list from the competition, and all_smiles.csv contains the smiles (ASCII string description of the molecules). Smiles were produced with pubchempy library. These to files are used in script.

We used these similarities combined with the descriptors (plus the square of both) as features and reduced the dimensionality with randomised Lasso (sklearn library). 



We have uploaded our Morgan feature set, which are Morgan similarities (uploaded the code as well that were used to calculate the features). The similarities were calculated not only between the 476 odorants, but include ~2000 other odor molecules as well. The idea was that some odors may share some common features with odorants outside of the set and may help the learning. Some of these molecules are known for their typical smell, like flower, or bread etc. 

Morgan fingerprints have very high dimensionality, they are not practical for use in learning in that form, but their similarities can be calculated, so that is included. The similarities between the 476 molecules only were already uploaded by Russ (mw.txt). 

There was one thing that improved our individual prediction a lot, maybe it is worth to give it a try. If I understood it correctly, some of you have combined the targets when predicted for the individuals, including the population average to some degree. 
We did something similar, but a modification helped for us: we calculated the correlation between individuals based on how they perceive a given smell across all the odors. For each individual, we calculated the weighted average of the population data and used this as target. The correlation with each individual was the weight. The correlation was 1 for the given person with its own data, but other's data were included as much as they had similar perception for the given descriptor. The calculations can be found in the hulab_prediction.py file in prediction_files folder.
Basically, the idea is to average the data for those who perceive the odors similarly.
