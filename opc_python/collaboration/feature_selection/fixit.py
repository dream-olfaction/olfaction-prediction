import pandas as pd

at = pd.read_csv("all_features_training_nonspdk.csv.tmp")
av = pd.read_csv("all_features_validation_nonspdk.csv.tmp")
nc = pd.read_csv("nspdk_gram_cid.csv.tmp")
merged_t = at.merge(nc, on='CID')
merged_t.to_csv("all_features_training_fixed.csv", index=False)
merged_v = av.merge(nc, on='CID')
merged_v.to_csv("all_features_validation_fixed.csv", index=False)

# The sum of the file sizes is not the same... 
# Some problem with missing data?
