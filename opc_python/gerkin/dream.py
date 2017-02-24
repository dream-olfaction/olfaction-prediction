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

def filter_Y_dilutions(df, concentration, keep_replicates=False):
    """Select only one dilution ('high' or 'low') and the mean across
    replicates for a given molecule."""
    assert concentration in ['high','low','gold'] or type(concentration) is int
    if keep_replicates:
        order = ['Descriptor','CID','Replicate']
    else:
        order = ['Descriptor','CID']
    df = df.sort_index(level=order+['Dilution'])
    if not keep_replicates:
        df = df.groupby(level=order+['Dilution']).mean() 
    df = df.fillna(999) # Pandas doesn't select correctly on NaNs
    if concentration == 'low':
        df = df.groupby(level=order).first()
    elif concentration == 'high':
        df = df.groupby(level=order).last()
    elif concentration == 'gold':
        a = df.loc[[x for x in df.index.values \
                    if x[0] == 'Intensity' and x[2]==-3]]
        a = a.groupby(level=order).last()
        b = df.drop('Intensity').groupby(level=order).last()
        df = pd.concat((a,b))
    else:
        df = df.loc[[x for x in df.index if x[2]==concentration]]
    df = df.replace(999,float('NaN')) # Undo the fillna line above. 
    # Get descriptors back in paper order
    descriptors = loading.get_descriptors(format=True)
    df = df.T[descriptors].T
    df['Subject'] = df['Subject'].astype(float) 
    return df


def impute(df,kind):
    if kind == 'median':
        imputer = Imputer(missing_values=np.nan,strategy='median',axis=0)
        df[:] = imputer.fit_transform(df)
    return df

############################
# Molecular processing (X) #
############################

def filter_X_dilutions(df, concentration):
    """Select only one dilution ('high', 'low', or some number)."""
    assert concentration in ['high','low'] or type(concentration) is int
    df = df.sort_index(level=['CID','Dilution']) 
    df = df.fillna(999) # Pandas doesn't select correctly on NaNs
    if concentration == 'low':
        df = df.groupby(level=['CID']).first()
    elif concentration == 'high':
        df = df.groupby(level=['CID']).last()
    else:
        df = df.loc[[x for x in df.index if x[1]==concentration]]
        df = df.groupby(level=['CID']).last()
    df = df.replace(999,float('NaN')) # Undo the fillna line above. 
    return df

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
        X,means,stds = normalize_X(X,means=means,stds=stds)
    else:
        good1,good2 = X.columns,X.columns
        means,stds,imputer = None,None,None
    if target_dilution is not None:
        X = filter_X_dilutions(X,target_dilution)

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
            molecular_vectors[CID] = np.array([np.nan if _=='NaN' else float(_)\
                                               for _ in row[1:]])
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
    # The X_obs matrix (molecular descriptors) still has NaN values that
    # need to be imputed.  
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

def normalize_X(X,means=None,stds=None):#,logs=None):
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

def quad_prep(mdx,CID_dilutions,dilution=None):
    """Given molecular data, return an array scaled between 0-1, 
    along with the squared versions of the same variables.  
    Put concentration information at the end of the array without
    squaring it.   
    """
    X,_,_,_,_,_ = make_X(mdx,CID_dilutions,target_dilution=dilution,
                              raw=True,quiet=True)
    X = X.fillna(0) 
    X_scaled = X.copy()
    X_scaled[:] = MinMaxScaler().fit_transform(X)
    X_scaled.drop(['dilution','mean_dilution'],1,inplace=True)
    X_scaled_sq = pd.concat((X_scaled,X_scaled**2,
                             X[['dilution','mean_dilution']]),1)
    print("The X matrix now has shape (%dx%d) molecules by " \
            % X_scaled_sq.shape + "non-NaN good molecular descriptors")
    
    return X_scaled_sq

#############
# Utilities #
#############

def purge(this,from_that):
    from_that = {CID:value for CID,value in from_that.items()\
                 if CID not in this}
    return from_that

def retain(this,in_that):
    in_that = {CID:value for CID,value in in_that.items()\
               if CID in this}
    return in_that
