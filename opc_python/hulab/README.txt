similarity_feature_calculation script calculates the similarities between the compounds using rdkit library. Test compound are compared with test compound and a set of other odorants. Odorant molecule list was collected from www.odour.org.uk and www.immuneweb.org/articles/fragrancelist.html.

Morgan fingerprints have very high dimensionality themselves, so only the similarities were used. CIDs.txt contains the odorant list from the competition, and all_smiles.csv contains the smiles (ASCII string description of the molecules). Smiles were produced with pubchempy library. These to files are used in script.

We used these similarities combined with the descriptors (plus the square of both) as features and reduced the dimensionality with randomised Lasso (sklearn library). 
