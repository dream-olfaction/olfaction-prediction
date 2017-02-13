import numpy as np
import pandas as pd
import types
from collections import OrderedDict
from sklearn.preprocessing import Imputer,MinMaxScaler

from opc_python import * # Import constants.  
from opc_python.utils import loading
DATA = '../../data/'

#############################
# Perceptual processing (Y) #
#############################

KINDS = ['training','training-norep','replicated',
         'leaderboard','testset','custom']

def make_Y_obs(kinds, target_dilution=None, imputer=None, quiet=False, subjects=range(1,50)):
    if target_dilution == 'gold':
        # For actual testing, use 1/1000 dilution for intensity and
        # high dilution for everything else.  
        Y,imputer = make_Y_obs(kinds,target_dilution='high',imputer=imputer,quiet=True)
        intensity,imputer = make_Y_obs(kinds,target_dilution=-3,imputer=imputer)
        Y['mean_std'][:,0] = intensity['mean_std'][:,0]
        Y['mean_std'][:,21] = intensity['mean_std'][:,21]
        for i in range(1,50):
            Y['subject'][i][:,0] = intensity['subject'][i][:,0]
        return Y,imputer
    if type(kinds) in [str,dict]:
        kinds = [kinds]
    if imputer in [None,'median']:
        imputer = Imputer(missing_values=np.nan,strategy='median',axis=0)
    Y = {}
    for i,kind in enumerate(kinds):
        if type(kind) is not dict:
            assert kind in KINDS, "No such kind %s" % kind
        if kind == 'leaderboard':
            loading.format_leaderboard_perceptual_data()
        if kind == 'testset':
            loading.format_testset_perceptual_data()
        if type(kind) is not dict:
            _, perceptual_data = loading.load_perceptual_data(kind)
            #print("Getting basic perceptual data...")
            matrices = get_perceptual_matrices(perceptual_data,
                                               target_dilution=target_dilution,
                                               subjects=subjects)
        else:
            matrices = kind
        #print("Flattening into vectors...")
        v_mean = get_perceptual_vectors(matrices, imputer=imputer, 
                                        statistic='mean', 
                                        target_dilution=target_dilution)
        v_std = get_perceptual_vectors(matrices, imputer=imputer, 
                                        statistic='std', 
                                        target_dilution=target_dilution)
        v_subject = get_perceptual_vectors(matrices, imputer=imputer, 
                                           statistic=None, 
                                           target_dilution=target_dilution)
        #print("Assembling into matrices...")
        Y[kind if type(kind) is str else i] = build_Y_obs(v_mean,v_std,v_subject)

    kinds = [kind if type(kind) is str else i for i,kind in enumerate(kinds)]
    #print("Combining Y matrices...")
    Y_ = {'subject':{}}
    Y_['mean_std'] = np.vstack([Y[kind]['mean_std'] for kind in kinds])#KINDS \
                                #if kind in kinds])
    for subject in subjects:
        Y_['subject'][subject] = np.ma.vstack([Y[kind]['subject'][subject] for kind in kinds])#
                                #KINDS if kind in kinds])
    if not quiet:
        print("The Y['mean_std'] matrix now has shape (%dx%d) " % Y_['mean_std'].shape +\
              "molecules x dilutions by 2 x perceptual descriptors")
        subject_data = list(Y_['subject'].values())
        print("The Y['subject'] dict now has %d matrices of shape (%dx%d) " % \
              (len(Y_['subject']),subject_data[0].shape[0],subject_data[0].shape[1]) +\
              "molecules x dilutions by perceptual descriptors, one for each subject")
    return Y_,imputer

def get_perceptual_vectors(perceptual_matrices, imputer=None, statistic='mean',
                           target_dilution=None):
    perceptual_vectors = {}
    for CID_dilution,matrix in perceptual_matrices.items():
        CID,dilution,high = CID_dilution.split('_')
        if target_dilution is None:
            pass
        elif target_dilution is 'low' and int(high)==1:
            continue
        elif target_dilution is 'high' and int(high)==0:
            continue
        elif type(target_dilution) is int \
            and target_dilution != int(dilution):
            continue
        matrix = matrix.copy()
        if imputer == 'zero':
            matrix[:,1][np.where(np.isnan(matrix[:,1]))] = 50 # Zero for pleasantness is 50.  
            matrix[:,2:][np.where(np.isnan(matrix[:,2:]))] = 0 # Zero for everything else is 0.  
        elif imputer == 'mask':
            mask = np.zeros(matrix.shape)
            mask[np.where(np.isnan(matrix))] = 1
            matrix = np.ma.array(matrix,mask=mask)
        elif imputer:
            matrix = imputer.fit_transform(matrix) # Impute the NaNs.
        if statistic == 'mean':
            perceptual_vectors[CID_dilution] = matrix.mean(axis=0)
        elif statistic == 'std':
            perceptual_vectors[CID_dilution] = matrix.std(axis=0,ddof=1)
        elif statistic is None:
            perceptual_vectors[CID_dilution] = {}
            for subject in range(1,matrix.shape[0]+1):        
                perceptual_vectors[CID_dilution][subject] = matrix[subject-1,:]
        else:
            raise Exception("Statistic '%s' not recognized" % statistic)
    return perceptual_vectors

def build_Y_obs(mean_vectors,std_vectors,subject_vectors,subjects=range(1,50)):
    Y = {'subject':{}}
    #x = [mean_vectors[CID] for CID in sorted(mean_vectors,key=lambda x:[int(_) for _ in x.split('_')])]
    #print([xi for xi in x])
    mean = np.vstack([mean_vectors[CID] for CID in sorted(mean_vectors,key=lambda x:[float(_) for _ in x.split('_')])])
    std = np.vstack([std_vectors[CID] for CID in sorted(std_vectors,key=lambda x:[float(_) for _ in x.split('_')])])
    for subject in subjects:
        Y['subject'][subject] = np.ma.vstack([subject_vectors[CID][subject]
                                               for CID in sorted(subject_vectors,key=lambda x:[float(_) for _ in x.split('_')])])
    #print("Y_obs['subject'] contains %d matrices each with shape (%dx%d) (molecules by perceptual descriptors)" \
    #  % (len(Y['subject']),Y['subject'][1].shape[0],Y['subject'][1].shape[1]))
    Y['mean_std'] = np.hstack((mean,std))
    #print("The Y_obs['mean_std'] matrix has shape (%dx%d) (molecules by 2 x perceptual descriptors)" % Y['mean_std'].shape)
    return Y

def nan_summary(perceptual_obs_matrices):
    z = np.dstack(perceptual_obs_matrices.values())

    nonzero_nonans,nonzero_nans,zero_nonans,zero_nonzero,zero_nans = 0,0,0,0,0
    for subject in range(49):
        for molecule in range(676):
            if z[subject,0,molecule]:
                nonzero_nonans += np.isnan(z[subject,1:,molecule]).sum()==0
                nonzero_nans += np.isnan(z[subject,1:,molecule]).sum()==20
            else:
                zero_nonans += np.isnan(z[subject,1:,molecule]).sum()==0
                zero_nonzero += z[subject,1:,molecule].max()>0
                zero_nans += np.isnan(z[subject,1:,molecule]).sum()==20
    print(nonzero_nonans,nonzero_nans,zero_nonans,zero_nonzero,zero_nans)

############################
# Molecular processing (X) #
############################

def make_X(df,CID_dilutions,target_dilution=None,threshold=None,bad=None,
           good1=None,good2=None,means=None,stds=None,raw=False,quiet=False):
    # df produced from e.g. loading.get_molecular_data()
    if threshold is None:
        threshold = NAN_PURGE_THRESHOLD
    data = [list(df.loc[CID])+[dilution,i] \
            for i,(CID,dilution) in enumerate(CID_dilutions)]
    X = pd.DataFrame(data=data,index=pd.MultiIndex.from_tuples(CID_dilutions,
                                     names=['CID','Dilution']),
                     columns=list(df.columns)+['dilution','mean_dilution'])
    if not raw:
        if bad:
            X = X.drop(bad)
        #print("Purging data with too many NaNs...")
        X,good1 = purge1_X(X,threshold=NAN_PURGE_THRESHOLD,
                           good_molecular_descriptors=good1)
        #print("Imputing remaining NaN data...")
        X,imputer = impute_X(X)
        #print("Purging data that is still bad, if any...")
        X,good2 = purge2_X(X,good_molecular_descriptors=good2)
        
        #print("Normalizing data for fitting...")
        X,means,stds = normalize_X(X,means=means,stds=stds,
                                   target_dilution=target_dilution)
    else:
        good1,good2 = X.columns,X.columns
        means,stds,imputer = None,None,None
    if not quiet:
        print("The X matrix now has shape (%dx%d) molecules by " % X.shape +\
          "non-NaN good molecular descriptors")
    return X,good1,good2,means,stds,imputer

def get_molecular_vectors(molecular_data,CID_dilutions):
    CIDs = []
    for CID_dilution in CID_dilutions:
        CID,dilution,high = CID_dilution.split('_')
        CIDs.append(int(CID)) 
    molecular_vectors = {}
    for row in molecular_data:
        CID = int(row[0])
        if CID in CIDs:
            molecular_vectors[CID] = np.array([np.nan if _=='NaN' else float(_) for _ in row[1:]])
    return molecular_vectors

'''
def add_dilutions(molecular_data,CID_dilutions,dilution=None):
    if 1:#dilution in [None,'low','high']:
        molecular_vectors_ = {}
        for CID_dilution in CID_dilutions:
            CID,dilution,high = [float(_) for _ in CID_dilution.split('_')]
            CID = int(CID); high = int(high)
            if high==1:
                mean_dilution = dilution - 1 # e.g. -3 was high so other was -5 so mean is -4.  
            elif high==0:
                mean_dilution = dilution + 1 # e.g. -3 was low so other was -1 so mean is -2. 
            else:
                raise ValueError("High not 0 or 1")
            molecular_vectors_[CID_dilution] = np.concatenate((molecular_vectors[CID],[dilution,mean_dilution]))
        molecular_vectors = molecular_vectors_
        #print('There are now %d molecular vectors of length %d, one for each molecule and dilution' \
        #    % (len(molecular_vectors),len(molecular_vectors[CID_dilution])))
    return molecular_vectors
'''

def purge1_X(X,threshold=0.25,good_molecular_descriptors=None):
    threshold = X.shape[0]*threshold
    if good_molecular_descriptors is None:
        # Which columns of X (which molecular descriptors) have NaNs for at 
        # least 'threshold' fraction of the molecules?  
        valid = np.isnan(X).sum()<threshold # True/False
        valid = valid[valid] # Only the 'Trues'
        good_molecular_descriptors = list(valid.index)
    X = X.loc[:,good_molecular_descriptors]
    #print("The X matrix has shape (%dx%d) (molecules by good molecular descriptors)" % X.shape)
    return X,good_molecular_descriptors

def impute_X(X):
    # The X_obs matrix (molecular descriptors) still has NaN values which need to be imputed.  
    imputer = Imputer(missing_values=np.nan,strategy='median',axis=0)
    X[:] = imputer.fit_transform(X)
    #print("The X matrix now has shape (%dx%d) (molecules by non-NaN good molecular descriptors)" % X.shape)
    return X,imputer

def purge2_X(X,good_molecular_descriptors=None):
    if good_molecular_descriptors is None:
        zeros = np.abs(np.sum(X,axis=0))==0 # All zeroes.  
        invariants = np.std(X,axis=0)==0 # Same value for all molecules.  
        valid = ~zeros & ~invariants
        valid = valid[valid]
        # Purge these bad descriptors from the X matrix.  
        good_molecular_descriptors = list(valid.index)
    X = X.loc[:,good_molecular_descriptors]
    #print("The X matrix has shape (%dx%d) (molecules by good molecular descriptors)" % X.shape)
    return X,good_molecular_descriptors

def normalize_X(X,means=None,stds=None,target_dilution=None):#,logs=None):
    num_cols = X.shape[1]
    num_cols -= 2
    X.iloc[:,:num_cols] = np.sign(X.iloc[:,:num_cols])*\
                            np.abs(X.iloc[:,:num_cols])**(1.0/3)
    if means is None:
        means = X.mean(axis=0)
        means[num_cols:] = 0
    if stds is None:
        stds = X.std(axis=0)
        stds[num_cols:] = 1
        stds[stds==0] = 1
    X = X.sub(means,axis=1)
    X = X.div(stds,axis=1)
    return X,means,stds

def quad_prep(mdx,sets=['training','leaderboard'],dilution=None):
    """Given molecular data, return an array scaled between 0-1, 
    along with the squared versions of the same variables.  
    Put concentration information at the end of the array without
    squaring it.  
    """
    X_temp,_,_,_,_,_ = make_X(mdx,sets,target_dilution=dilution,
                              raw=True,quiet=True)
    X_temp[np.isnan(X_temp)] = 0     
    X_scaled = MinMaxScaler().fit_transform(X_temp[:,:-2])
    X_scaled_sq = np.hstack((X_scaled,X_scaled**2,X_temp[:,-2:]))
    print("The X matrix now has shape (%dx%d) molecules by " \
            % X_scaled_sq.shape + "non-NaN good molecular descriptors")
    
    return X_scaled_sq

#############
# Utilities #
#############

def purge(this,from_that):
    from_that = {CID:value for CID,value in from_that.items() if CID not in this}
    return from_that

def retain(this,in_that):
    in_that = {CID:value for CID,value in in_that.items() if CID in this}
    return in_that
