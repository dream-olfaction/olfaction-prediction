
# coding: utf-8

# In[2]:

# this script creates csvs with target data
# replace plesantness with Nan where intensity is zero
# then average all the remaining samples for each compound
# replace descriptors with Nan if intensity is still zero somewhere
# save them in CSV as avg_targets.csv


# In[3]:

import pandas as pd
import numpy as np


# In[5]:


# load the train data 
data = pd.read_csv("TrainSet.txt",sep='\t')
# average the duplicates
data = data.groupby(['subject #','Compound Identifier','Intensity','Dilution']).mean() #need this to average the replicates
data.reset_index(level=[1,2,3,0], inplace=True)


# In[6]:

# we don't need the intensity here
data.drop(['Intensity'],axis=1, inplace=1)


# In[7]:

data.columns = [u'individual', u'#oID','Dilution', u'INTENSITY/STRENGTH', u'VALENCE/PLEASANTNESS', u'BAKERY', u'SWEET', u'FRUIT', u'FISH', u'GARLIC', u'SPICES', u'COLD', u'SOUR', u'BURNT', u'ACID', u'WARM', u'MUSKY', u'SWEATY', u'AMMONIA/URINOUS', u'DECAYED', u'WOOD', u'GRASS', u'FLOWER', u'CHEMICAL']


# In[8]:

# replace descriptors with Nan where intensity is zero
data.loc[data['INTENSITY/STRENGTH'] ==0,[u'VALENCE/PLEASANTNESS', u'BAKERY', u'SWEET', u'FRUIT', u'FISH', u'GARLIC', u'SPICES', u'COLD', u'SOUR', u'BURNT', u'ACID', u'WARM', u'MUSKY', u'SWEATY', u'AMMONIA/URINOUS', u'DECAYED', u'WOOD', u'GRASS', u'FLOWER', u'CHEMICAL']] = np.nan


# In[9]:

#average the data

data_avg = data.groupby('#oID').mean() 
data_avg.drop('individual',axis=1,inplace=1)


# In[10]:

# average intensity data in duplicates

data_int = data[data.Dilution == '1/1,000 '].groupby('#oID').mean() 
data_int.drop('individual',axis=1,inplace=1)


# In[12]:

data_avg['INTENSITY/STRENGTH'] = data_int['INTENSITY/STRENGTH']


# In[12]:

data_avg.to_csv('targets_for_feature_selection.csv')


# In[ ]:



