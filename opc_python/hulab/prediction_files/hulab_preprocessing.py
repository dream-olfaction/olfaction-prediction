### DREAM Olfaction Prediction Challenge
### Created by: Bence Szalai and Gabor Turu
### Contact: bence.szalai@eok.sote.hu & gabor.turu@eok.sote.hu
import os
import numpy as np
import pandas as pd

### Change these variables to give input data/path
DIRECTORY='/media/szalaib/D60C49380C491541/Data/Dry/DREAM/2015_olfactory/final'
TRAIN='TrainSet.txt'
LEADER='LBs1.txt'

os.chdir(DIRECTORY)

### call run_data_preprocessing() to do the data preprocessing

def run_data_preprocessing(train=TRAIN,leader=LEADER):
    """
    performs data preprocessing
    """
    print 'Creating data files...'
    create_input_data(train,leader)
    print 'Calculating population means...'
    create_population_means_and_stds()
    print 'Calculation individual means and stds for descriptors...'
    create_individual_means_and_stds()
    print 'Ready!'
    
def create_input_data(train,leader):
    """
    Creates data files from trainset and leaderboard data
    Created files have the format D_S.txt where D is a descriptor and S is a subject
    """
    ### reads data from trainset
    fin=open(train,'r')
    temp=fin.readlines()
    fin.close()
    flines=temp[0].split('\r')
    descriptors=['Intensity','Valence','Bakery','Sweet','Fruit','Fish','Garlic','Spices','Cold','Sour','Burnt','Acid','Warm','Musky','Sweaty','Ammonia','Decayed','Wood','Grass','Flower','Chemical']
    descriptors_original=['INTENSITY/STRENGTH','VALENCE/PLEASANTNESS','BAKERY','SWEET','FRUIT','FISH','GARLIC','SPICES','COLD','SOUR','BURNT','ACID','WARM','MUSKY','SWEATY','AMMONIA/URINOUS','DECAYED','WOOD','GRASS','FLOWER',' CHEMICAL']
    data={}
    for descriptor in descriptors:
        data[descriptor]={}
        for subject in range(1,50):
            data[descriptor][str(subject)]={}
    for i in range(1,len(flines)):
        line=flines[i].split('\t')
        cid=line[0]
        subject=line[5]
        if line[2]!='replicate ':
            if line[4]=='"1/1,000 "':
                data['Intensity'][subject][cid]=line[6]
            if line[3]=='high ':
                for j in range(7,len(line)):
                    if line[j]!='NaN':
                        data[descriptors[j-6]][subject][cid]=line[j]
    ### reads data from leaderboard
    fin=open(leader,'r')
    temp=fin.readlines()
    fin.close()
    flines=temp[0].split('\r')
    for i in range(1,len(flines)):
        line=flines[i].split('\t')
        cid=line[0]
        subject=line[1]
        descriptor_original=line[2]
        descriptor=descriptors[descriptors_original.index(descriptor_original)]
        value=line[3]
        if value!='NaN':
            data[descriptor][subject][cid]=value
    for descriptor in descriptors:
        for subject in range(1,50):
            fout=open(descriptor+'_'+str(subject)+'.txt','w')
            fout.write('CID\tValue\n')
            for cid in data[descriptor][str(subject)]:
                fout.write(cid+'\t'+data[descriptor][str(subject)][cid]+'\n')
            fout.close()
            
def create_population_means_and_stds():
    """
    creates population mean and std for descriptors
    Created files have the format D_0.txt where D is a descriptor
    """
    descriptors=['Intensity','Valence','Bakery','Sweet','Fruit','Fish','Garlic','Spices','Cold','Sour','Burnt','Acid','Warm','Musky','Sweaty','Ammonia','Decayed','Wood','Grass','Flower','Chemical']
    data={}
    for descriptor in descriptors:        
        data[descriptor]={}
        for subject in range(1,50):
            fin=open(descriptor+'_'+str(subject)+'.txt','r')
            flines=fin.readlines()
            fin.close()
            for i in range(1,len(flines)):
                line=flines[i].split('\t')
                try:
                    data[descriptor][line[0]].append(float(line[1]))
                except KeyError:
                    data[descriptor][line[0]]=[float(line[1])]
    for descriptor in descriptors:
        fout=open(descriptor+'_0.txt','w')
        fout.write('CID\tValue\tSD\n')
        for cid in data[descriptor]:
            fout.write(cid+'\t'+str(np.mean(data[descriptor][cid]))+'\t'+str(np.std(data[descriptor][cid]))+'\n')
        fout.close()
        
def read_data(subject,descriptor):
    """
    reads data into a pandas DataFrame
    """
    data=pd.read_table(descriptor+'_'+str(subject)+'.txt',sep='\t',header=0,index_col='CID')
    return data.loc[:,'Value']

def create_individual_means_and_stds():
    """
    creates individual mean and std for the different descriptors
    """
    fout=open('means_stds.txt','w')
    fout.write('Subject\tDescriptor\tValue\tSD\n')
    descriptors=['Intensity','Valence','Bakery','Sweet','Fruit','Fish','Garlic','Spices','Cold','Sour','Burnt','Acid','Warm','Musky','Sweaty','Ammonia','Decayed','Wood','Grass','Flower','Chemical']
    for subject in range(0,50):
        for descriptor in descriptors:
            data=read_data(subject,descriptor)
            if sum(data.values!=0)!=0:
                fout.write(str(subject)+'\t'+descriptor+'\t'+str(np.mean(data.values))+'\t'+str(np.std(data.values))+'\n')
            else:
                ### if all data points are 0, set standard deviation to 1
                fout.write(str(subject)+'\t'+descriptor+'\t'+str(0)+'\t'+str(1)+'\n')
    fout.close()
    
