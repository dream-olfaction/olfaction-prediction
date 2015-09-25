Readme for meansx.csv, DREAM Olfaction data

The file contains two rows for each molecule, corresponding to two different measured dilutions.  The prediction
features are duplicated for each pair of rows.

Columns:

CID, Compound Identifier
source,  train (338 molecules), leaderboard(69 molecules), test (69 molecules)
smiles, smiles string
neglog10d,  numeric coding of log10 dilution
dilution, dilution as a string
intensity,  high/low value based on relative comparison of row pairs
intensity_strength through chemical,  means of measured targets
beginning with f_,  Dragon features, range normalized to be between 0 and 1
beginning with johnson or none,  Episuite features, the johnson ones are transformed with johnson to normalize
beginning with col, square root of Morgan feature weights gram matrix
beginning with nspdk, square root of NSPDK gram matrix
beginning with abag, square root of ABAG gram matrix
beginning with name, square root of Name feature gram matrix
