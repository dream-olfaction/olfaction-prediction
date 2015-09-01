
# coding: utf-8

# In[1]:

# this script creates csvs with target data
# replace plesantness with Nan where intensity is zero
# then average all the remaining samples for each compound
# replace descriptors with Nan if intensity is still zero somewhere
# save them in CSV as avg_targets.csv


# In[2]:

import pandas as pd
import numpy as np


# In[3]:


# load the train data 
data = pd.read_csv("TrainSet.txt",sep='\t')
# average the duplicates
data = data.groupby(['subject #','Compound Identifier','Intensity','Dilution']).mean() #need this to average the replicates
data.reset_index(level=[1,2,3,0], inplace=True)


# In[4]:

# features for all except INTENSITY/STRENGTH will be selected with combined low/high data 
# no need for Intensity
data.drop(['Intensity'],axis=1, inplace=1) 


# In[5]:

# rename to match the LB dataset column names
data.columns = [u'individual', u'#oID','Dilution', u'INTENSITY/STRENGTH', u'VALENCE/PLEASANTNESS', u'BAKERY', u'SWEET', u'FRUIT', u'FISH', u'GARLIC', u'SPICES', u'COLD', u'SOUR', u'BURNT', u'ACID', u'WARM', u'MUSKY', u'SWEATY', u'AMMONIA/URINOUS', u'DECAYED', u'WOOD', u'GRASS', u'FLOWER', u'CHEMICAL']


# In[6]:

# load LB data, reshape them to match the train data
LB_data_high = pd.read_csv("LBs1.txt",sep='\t')
LB_data_high = LB_data_high.pivot_table(index=['#oID','individual'],columns='descriptor',values='value')
LB_data_high.reset_index(level=[0,1],inplace=1)
LB_data_high.columns = [u'#oID', u'individual', u'CHEMICAL', u'ACID', u'AMMONIA/URINOUS', u'BAKERY', u'BURNT', u'COLD', u'DECAYED', u'FISH', u'FLOWER', u'FRUIT', u'GARLIC', u'GRASS', u'INTENSITY/STRENGTH', u'MUSKY', u'SOUR', u'SPICES', u'SWEATY', u'SWEET', u'VALENCE/PLEASANTNESS', u'WARM', u'WOOD']
LB_data_high['Dilution'] = '1/1,000 ' # I guess these are all diluted 1/1000
LB_data_high = LB_data_high[data.columns]


# In[7]:

LB_data_low = pd.read_csv("leaderboard_set_Low_Intensity.txt",sep='\t')
LB_data_low = LB_data_low.pivot_table(index=['#oID','individual'],columns='descriptor',values='value')
LB_data_low.reset_index(level=[0,1],inplace=1)
LB_data_low.columns = [u'#oID', u'individual', u'CHEMICAL', u'ACID', u'AMMONIA/URINOUS', u'BAKERY', u'BURNT', u'COLD', u'DECAYED', u'FISH', u'FLOWER', u'FRUIT', u'GARLIC', u'GRASS', u'INTENSITY/STRENGTH', u'MUSKY', u'SOUR', u'SPICES', u'SWEATY', u'SWEET', u'VALENCE/PLEASANTNESS', u'WARM', u'WOOD']
LB_data_low['Dilution'] = ' ' # just adding Dilution column
LB_data_low = LB_data_low[data.columns]


# In[8]:

# putting them all together
data = pd.concat((data,LB_data_high,LB_data_low),ignore_index=True)


# In[9]:

# replace descriptors with Nan where intensity is zero
data.loc[data['INTENSITY/STRENGTH'] ==0,[u'VALENCE/PLEASANTNESS', u'BAKERY', u'SWEET', u'FRUIT', u'FISH', u'GARLIC', u'SPICES', u'COLD', u'SOUR', u'BURNT', u'ACID', u'WARM', u'MUSKY', u'SWEATY', u'AMMONIA/URINOUS', u'DECAYED', u'WOOD', u'GRASS', u'FLOWER', u'CHEMICAL']] = np.nan


# In[10]:

#average the data
data_avg = data.groupby('#oID').mean() 
data_avg.drop('individual',axis=1,inplace=1)


# In[11]:

# average 1/1000 dilutions in separate DF ...
data_int = data[data.Dilution == '1/1,000 '].groupby('#oID').mean() 
data_int.drop('individual',axis=1,inplace=1)


# In[12]:

# ... then use the INTENSITY/STRENGTH data
data_avg['INTENSITY/STRENGTH'] = data_int['INTENSITY/STRENGTH']


# In[31]:

# save it
data_avg.to_csv('targets_for_feature_selection_LB_incl.csv')


# In[ ]:



