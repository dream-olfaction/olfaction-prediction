
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from scipy import stats as stats
import os


DESCRIPTORS=['INTENSITY/STRENGTH','VALENCE/PLEASANTNESS','BAKERY','SWEET','FRUIT','FISH','GARLIC','SPICES','COLD','SOUR','BURNT','ACID','WARM','MUSKY','SWEATY','AMMONIA/URINOUS','DECAYED','WOOD','GRASS','FLOWER','CHEMICAL']
SUBJECTS=range(1,50)

def read_data(fname):
    data=pd.read_table(fname,sep='\t',header=0,index_col='#oID')
    return data
    
def calculate_correlations_2():
    global DESCRIPTORS
    global SUBJECTS
    r={}
    s={}

    data1=read_data(os.path.abspath('__file__' + "/../../../data/LBs2.txt"))
    data2=read_data('subchallenge2.txt')
    
    data1 = data1.sort().sort('descriptor')
    data1.descriptor[data1.descriptor== ' CHEMICAL'] = 'CHEMICAL'
    data1_mean = data1.reset_index().pivot_table(index = '#oID', columns = 'descriptor', values='value')
    data1_std = data1.reset_index().pivot_table(index = '#oID', columns = 'descriptor', values='sigma')
    data2 = data2.sort().sort('descriptor')
    data2_mean = data2.reset_index().pivot_table(index = '#oID', columns = 'descriptor', values='value')
    data2_std = data2.reset_index().pivot_table(index = '#oID', columns = 'descriptor', values='sigma')
    scores = []
    scores.append(stats.pearsonr(data1_mean['INTENSITY/STRENGTH'],data2_mean['INTENSITY/STRENGTH'])[0])
    scores.append(stats.pearsonr(data1_mean['VALENCE/PLEASANTNESS'],data2_mean['VALENCE/PLEASANTNESS'])[0])

    others = [] 
    for desc in data1_std.columns:
        print desc + ':' + str(stats.pearsonr(data1_mean[desc],data2_mean[desc])[0])
        if desc not in ['INTENSITY/STRENGTH','VALENCE/PLEASANTNESS']:
            others.append(stats.pearsonr(data1_mean[desc],data2_mean[desc])[0])
            
    scores.append(np.mean(others))  

    scores.append(stats.pearsonr(data1_std['INTENSITY/STRENGTH'],data2_std['INTENSITY/STRENGTH'])[0])
    scores.append(stats.pearsonr(data1_std['VALENCE/PLEASANTNESS'],data2_std['VALENCE/PLEASANTNESS'])[0])

    others = [] 
    for desc in data1_std.columns:
        
        if desc not in ['INTENSITY/STRENGTH','VALENCE/PLEASANTNESS']:
            others.append(stats.pearsonr(data1_std[desc],data2_std[desc])[0])
    scores.append(np.mean(others)) 

    sigmas=np.array([0.1193,0.1265,0.0265,0.1194,0.1149,0.0281])
    return np.round(scores,3), np.round(sum(scores/sigmas)/6,3)




print(calculate_correlations_2())



