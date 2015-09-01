
# coding: utf-8

# #### this script calculates similarity features, combines them with descriptor data and selects the best features with randomized Lasso


# In[1]:

import pandas as pd
import numpy as np
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors as Descriptors
from sklearn import preprocessing
from sklearn.linear_model import RandomizedLasso
import sys


# In[2]:

# get train, leaderboard and test CIDs
with open('CIDs.txt') as f: 
    content = f.readlines()
CIDs = list(content)  
CIDs = [int(x) for x in CIDs]

# get smiles
smiles = pd.read_csv('all_smiles.csv', index_col=0) # load smiles if the file already exists


# In[3]:

# function to calculate the features from Morgan fingerprints
# creates the fingerprints and calculates similarities 
# Inputs: 
#       list of ids
#       Morgan radius
# Returns:
#       feature vector with size of len(cids) x number of features


def calulate_similarities(ids, radius):
    ms = [Chem.MolFromSmiles(x) for x in smiles.smiles]
    fps = [AllChem.GetMorganFingerprint(x,radius) for x in ms]
    all_features =[]
    for idx, cid in enumerate(ids):
        ms_sample = Chem.MolFromSmiles(smiles.loc[cid].smiles)
        fp_sample = AllChem.GetMorganFingerprint(ms_sample,radius)
        features = [cid]
        for fp in fps:
            features.append(DataStructs.DiceSimilarity(fp,fp_sample))
        print(idx,end='\r')
        all_features.append(features)
    all_features = pd.DataFrame(all_features)
    all_features = all_features.set_index(0)
    all_features.columns = smiles.index
    return all_features    


# In[4]:

# load the feature descriptors, scale them, add square
descriptors =pd.read_csv('molecular_descriptors_data.txt', sep='\t')
descriptors.set_index('CID', inplace=True)
descriptors.sort(inplace=True)
descriptors.fillna(value=0,inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
descriptors.ix[:,:]= min_max_scaler.fit_transform(descriptors)
descriptors = pd.concat((descriptors,descriptors**2),axis=1) # add squares
print(descriptors.shape)


# In[5]:

# get the similarity features, add square
features_sim = calulate_similarities(descriptors.index, 5)
features_sim = pd.concat((features_sim,features_sim**2), axis = 1) # add squares
print(features_sim.shape)


# In[6]:

# put them together

all_features = pd.concat((features_sim,descriptors),axis=1)
all_features.shape


# In[8]:

#print (all_features.head())


# In[9]:

# save it 
all_features.to_csv('all_features.csv')


# In[10]:

# pair descriptors with numbers
attribute = {}
for idx, attr in enumerate([u'INTENSITY/STRENGTH', u'VALENCE/PLEASANTNESS', u'BAKERY', 
                       u'SWEET', u'FRUIT', u'FISH', u'GARLIC', u'SPICES', u'COLD', u'SOUR', u'BURNT',
                       u'ACID', u'WARM', u'MUSKY', u'SWEATY', u'AMMONIA/URINOUS', u'DECAYED', u'WOOD',
                       u'GRASS', u'FLOWER', u'CHEMICAL']):
    attribute[idx] = attr


# In[ ]:

# select  the best features with true values and save them
features = pd.read_csv('all_features.csv',index_col=0).sort()
target = pd.read_csv('targets_for_feature_selection.csv',index_col=0).sort()#replace this with targets_for_feature_selection_LB_incl.csv if LB data is included
for i in range(21):
    print(attribute[i])
    sys.stdout.flush()
    
    
    Y = target[attribute[i]].dropna()
    X = features.loc[Y.index]
    selector = RandomizedLasso(alpha=0.025,selection_threshold=0.025,n_resampling=200,
                               random_state=25).fit(X,Y)
    selected = pd.DataFrame(selector.transform(features))
    selected.index = features.index
    print('shape ', selected.shape)
    
    selected.to_csv('...path to features folder/selected_features/features_'+str(i)+'.csv')


# In[ ]:



