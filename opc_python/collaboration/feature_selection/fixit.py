#/usr/bin/python3

import pandas as pd

train = pd.read_csv("all_features_training.csv")
valid = pd.read_csv("all_features_validation.csv")

if train.columns[[7307]] != 'nspdk1':
  raise Exception('The input file doesn not seem to contain the error. Maybe it has already been fixed?')

# delete the permuted nspdk columns
train=train.drop(train.columns[list(range(7307,9744))], axis=1)
valid=valid.drop(valid.columns[list(range(7307,9744))], axis=1)

# add correct nspdk columns
# (Note that these are NPSDK kernel values, i.e. similarities with other molecules, not NPSDK features.)
nc = pd.read_csv("nspdk_gram_cid.csv.tmp")
train = train.merge(nc, on='CID')
valid = valid.merge(nc, on='CID')

train.to_csv("all_features_training_fixed.csv", index=False)
valid.to_csv("all_features_validation_fixed.csv", index=False)
