### DREAM Olfaction Prediction Challenge
### Created by: Bence Szalai and Gabor Turu
### Contact: bence.szalai@eok.sote.hu & gabor.turu@eok.sote.hu
import numpy as np
from scipy import stats as stats
import pandas as pd
import os
from sklearn import linear_model as lm

### Change these variable to give input path
DIRECTORY='/media/szalaib/D60C49380C491541/Data/Dry/DREAM/2015_olfactory/final'

os.chdir(DIRECTORY)

### call run_predictions() to do the predictions for the two subchallenge

def run_predictions():
    """
    performs predictions
    """
    subchallenge1()
    subchallenge2()
    
def read_data_tr(subject,descriptor):
    """
    reads training data into a pandas DataFrame
    """
    data=pd.read_table(descriptor+'_'+str(subject)+'.txt',sep='\t',header=0,index_col='CID')
    return data.loc[:,'Value']
    
def read_data_md(descriptor):
    """
    reads features into a pandas DataFrame
    descriptor is an integer 0..20, where 0 is Intensity, 1 is Valence, 2 is Bakery etc.
    """
    mol_des=pd.read_table('features_'+str(descriptor)+'.csv',sep=',', header=0, index_col='CID')
    return mol_des
    
def read_weights():
    """
    reads the Morgan similarities data to weight the training samples
    """
    weights=pd.read_table('weights.csv',sep=',',header=0, index_col='0')
    return weights
    
def read_individual_means_stds():
    means_stds=pd.read_table('means_stds.txt',sep='\t',header=0,index_col=None)
    return means_stds
    
def predict_for_individual(subject,descriptor,mol_desc,cids_tr,cids_pr):
    """
    fit linear model and make predictions
    """
    LinearModel=lm.Ridge(0.5)
    data=read_data_tr(subject,descriptor)
    data=data[cids_tr]
    s=subject
    if sum(data!=0)==0:
        ### if all values are 0, make predictions based on the population mean
        data_to_use=read_data_tr(0,descriptor)
        s=0
    else:
	### this part calculates the targets for individuals from population data
        ### calculate the weighted average for descriptor, based on the
        ### similarities between individuals for the given descriptor
        data_to_use=pd.Series(np.zeros(np.shape(data)),index=data.index)
        data_counts=pd.Series(np.zeros(np.shape(data)),index=data.index)
        for subject2 in range(1,50):
            data2=read_data_tr(subject2,descriptor)
            indexes=set(data.index)&set(data2.index)
            ### calculates the correlation between subject and subject2
            r=stats.pearsonr(data[indexes],data2[indexes])[0]
            if not np.isnan(r):
                data_to_use[indexes]+=data2[indexes]*r
                data_counts[indexes]+=r
        data_to_use=data_to_use/data_counts
    ### y is training vector
    y=data_to_use[cids_tr]
    ### x is the features matrix
    x=mol_desc.loc[cids_tr,:]
    ### x_pr is features matrix for prediction
    x_pr=mol_desc.loc[cids_pr,:]
    y_pr=[]
    ### reading the Morgan similarities for weighting the training examples
    weights=read_weights()
    if descriptor not in ['Intensity','Valence']:
        ### weighting is only used for the other 19 descriptors
        for i in range(len(x_pr.index)):
            weight=weights.loc[x.index,str(x_pr.index[i])]
            if sum(weight!=0)>0:
                LinearModel.fit(x,y,sample_weight=weight)
                temp_pr=LinearModel.predict(x_pr)
                temp_pr=temp_pr[i]
                y_pr.append(temp_pr)
            else:
                ### if all weights are 0, do not use weights
                LinearModel.fit(x,y)
                temp_pr=LinearModel.predict(x_pr)
                temp_pr=temp_pr[i]
                y_pr.append(temp_pr)    
        y_pr=np.array(y_pr)
    else:
        ### if descriptor is Intensity of Valence, do not use weights
        LinearModel.fit(x,y)
        y_pr=LinearModel.predict(x_pr)
    if sum(y_pr!=0)!=0:
        ### normalize the distribution of predictions back to the original ditribution of training data
        y_pr=(y_pr-np.mean(y_pr))/np.std(y_pr)
        mean_std=read_individual_means_stds()
        fil1=mean_std.loc[:,'Descriptor']==descriptor
        fil2=mean_std.loc[:,'Subject']==s
        fil=fil1&fil2
        y_pr=y_pr*np.array(mean_std[fil].loc[:,'SD'])+np.array(mean_std[fil].loc[:,'Value'])
    ### if predicted values are larger than 100 or smaller than 0, set them to 100/0
    fil1=y_pr>100
    fil0=y_pr<0
    y_pr[fil1]=100
    y_pr[fil0]=0
    return y_pr
    
def read_CIDs():
    """
    reads CIDs for final
    """
    fin=open('CIDs_final.txt','r')
    flines=fin.readlines()
    fin.close()
    CIDs=[]
    for line in flines:
        if line[-1]=='\n':
            CIDs.append(int(line[:-1]))
        else:
            CIDs.append(int(line))
    return CIDs

def subchallenge1():
    """
    runs predictions for subchallenge1
    """
    print 'Predictions for subchellenge1...'
    descriptors=['Intensity','Valence','Bakery','Sweet','Fruit','Fish','Garlic','Spices','Cold','Sour','Burnt','Acid','Warm','Musky','Sweaty','Ammonia','Decayed','Wood','Grass','Flower','Chemical']
    descriptors_good=['INTENSITY/STRENGTH','VALENCE/PLEASANTNESS','BAKERY','SWEET','FRUIT','FISH','GARLIC','SPICES','COLD','SOUR','BURNT','ACID','WARM','MUSKY','SWEATY','AMMONIA/URINOUS','DECAYED','WOOD','GRASS','FLOWER','CHEMICAL']
    cids_pr=read_CIDs()
    fout=open('subchallenge1.txt','w')
    fout.write('#oID\tindividual\tdescriptor\tvalue\n')
    for subject in range(1,50):
        print 'Running prediction for subject '+str(subject)+'...'
        for i in range(len(descriptors)):
            mol_desc=read_data_md(i)
            data=read_data_tr(subject,descriptors[i])
            cids_tr=data.index
            y_pr=predict_for_individual(subject,descriptors[i],mol_desc,cids_tr,cids_pr)
            for j in range(len(cids_pr)):
                fout.write(str(cids_pr[j])+'\t'+str(subject)+'\t'+descriptors_good[i]+'\t'+str(y_pr[j])+'\n')
    fout.close()

def subchallenge2():
    """
    runs predictions for subchallenge2
    """
    print 'Predictions for subchellenge2...'
    fin=open('subchallenge1.txt','r')
    flines=fin.readlines()
    fin.close()
    sub_dict={}
    for i in range(1,len(flines)):
        line=flines[i].split()
        if line[0] not in sub_dict.keys():
            sub_dict[line[0]]={}
        if line[2] not in sub_dict[line[0]].keys():
            sub_dict[line[0]][line[2]]=[]
        sub_dict[line[0]][line[2]].append(float(line[3]))
    fout=open('subchallenge2.txt','w')
    fout.write('#oID\tdescriptor\tvalue\tsigma\n')
    for cid in sub_dict.keys():
        for desc in sub_dict[cid]:
            ### calculates predicted mean
            m=np.mean(sub_dict[cid][desc])
            ### calculates predicted std from mean
            s=-(m**2)/2500.0+m/25.0
            fout.write(str(cid)+'\t'+desc+'\t'+str(m)+'\t'+str(s)+'\n')
    fout.close()


