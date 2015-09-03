#from __future__ import print_function
# coding: utf-8

# this script selects the best features with randomized Lasso

import pandas as pd
import numpy as np


from sklearn import preprocessing
from sklearn.linear_model import RandomizedLasso
import sys

# replace messing values with 0 (Gabor's approach)
descriptors =pd.read_csv('all_features.csv', sep=',')
descriptors.set_index('CID', inplace=True)
descriptors.fillna(value=0,inplace=True)
descriptors.to_csv('all_features_filledna.csv')

# pair descriptors with numbers
attribute = {}
for idx, attr in enumerate([u'INTENSITY/STRENGTH', u'VALENCE/PLEASANTNESS', u'BAKERY', 
                       u'SWEET', u'FRUIT', u'FISH', u'GARLIC', u'SPICES', u'COLD', u'SOUR', u'BURNT',
                       u'ACID', u'WARM', u'MUSKY', u'SWEATY', u'AMMONIA/URINOUS', u'DECAYED', u'WOOD',
                       u'GRASS', u'FLOWER', u'CHEMICAL']):
    attribute[idx] = attr



# select the best features with true values and save them
features = pd.read_csv('all_features_filledna.csv',index_col=0).sort()
# targets_for_feature_selection.csv can be computed in opc_python/hulab/feature_selection
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
    
    selected.columns = X.ix[:,selector.scores_>0.025].columns

    f = open('scores_'+str(i)+'.txt', 'w')
    #nb_features = int(features.length())
    for x in range(0, len(features.columns.values)):
      feature_name = features.columns.values[x]
      f.write(feature_name+': ')
      score = selector.scores_[x]
      f.write(str(score))
      f.write('\n')

    f.close()

    #selected.to_csv('features_'+str(i)+'.csv')




