import numpy as np
import types
from collections import OrderedDict
from sklearn.preprocessing import Imputer

from opc_python import * # Import constants.  
from opc_python.utils import loading

#############################
# Perceptual processing (Y) #
#############################

def make_Y_obs(kinds, target_dilution=None, imputer=None, quiet=False):
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
    if type(kinds) is str:
        kinds = [kinds]
    if imputer in [None,'median']:
        imputer = Imputer(missing_values=np.nan,strategy='median',axis=0)
    Y = {}
    for kind in kinds:
        assert kind in ['training','leaderboard','testset'], \
            "No such kind %s" % kind
        if kind == 'leaderboard':
            loading.format_leaderboard_perceptual_data()
        _, perceptual_data = loading.load_perceptual_data(kind)
        #print("Getting basic perceptual data...")
        matrices = get_perceptual_matrices(perceptual_data,
                                            target_dilution=target_dilution)
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
        Y[kind] = build_Y_obs(v_mean,v_std,v_subject)

    #print("Combining Y matrices...")
    Y_ = {'subject':{}}
    Y_['mean_std'] = np.vstack([Y[kind]['mean_std'] for kind in 
                                ['training','leaderboard','testset'] \
                                if kind in kinds])
    for subject in range(1,50):
        Y_['subject'][subject] = np.ma.vstack([Y[kind]['subject'][subject] for kind in 
                                ['training','leaderboard','testset'] \
                                if kind in kinds])
    if not quiet:
        print("The Y['mean_std'] matrix now has shape (%dx%d) " % Y_['mean_std'].shape +\
              "molecules by 2 x perceptual descriptors")
        print("The Y['subject'] dict now has %d matrices of shape (%dx%d) " % \
              (len(Y_['subject']),Y_['subject'][1].shape[0],Y_['subject'][1].shape[1]) +\
              "molecules by perceptual descriptors, one for each subject")
    return Y_,imputer

def get_perceptual_matrices(perceptual_data,target_dilution=None,use_replicates=True):
    perceptual_matrices = {}
    counts = {}
    CIDs = []
    for row in perceptual_data:
        CID = int(row[0])
        replicate = int(row[2])
        CIDs.append(CID)
        dilution = loading.dilution2magnitude(row[4])
        if target_dilution is None:
            pass
        elif dilution == target_dilution:
            pass
        elif target_dilution in ['low','high']:
            if row[3] != target_dilution:
                continue
        else:
            continue
        high = row[3] == 'high'
        key = '%d_%g_%d' % (CID,dilution,high)
        if key not in perceptual_matrices:
            perceptual_matrices[key] = np.ones((49,21))*np.NaN
            counts[key] = np.zeros(49)
        data = np.array([np.nan if _=='NaN' else int(_) for _ in row[6:]])
        subject = int(row[5])
        if replicate:
            if use_replicates:
                perceptual_matrices[key][subject-1,:] *= counts[key][subject-1]
                perceptual_matrices[key][subject-1,:] += data
                counts[key][subject-1] += 1
                perceptual_matrices[key][subject-1,:] /= counts[key][subject-1]
        else:
            perceptual_matrices[key][subject-1,:] = data
                
    return perceptual_matrices

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

def build_Y_obs(mean_vectors,std_vectors,subject_vectors):
    Y = {'subject':{}}
    mean = np.vstack([mean_vectors[CID] for CID in sorted(mean_vectors,key=lambda x:[int(_) for _ in x.split('_')])])
    std = np.vstack([std_vectors[CID] for CID in sorted(std_vectors,key=lambda x:[int(_) for _ in x.split('_')])])
    for subject in range(1,50):
        Y['subject'][subject] = np.ma.vstack([subject_vectors[CID][subject]
                                               for CID in sorted(subject_vectors,key=lambda x:[int(_) for _ in x.split('_')])])
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

def make_X(molecular_data,kinds,target_dilution=None,threshold=None,
           good1=None,good2=None,means=None,stds=None):
    if type(kinds) is str:
        kinds = [kinds]
    if threshold is None:
        threshold = NAN_PURGE_THRESHOLD
    #print("Getting CIDs and dilutions...")
    CID_dilutions = []
    for kind in kinds:
        assert kind in ['training','leaderboard','testset'], \
            "No such kind %s" % kind
        CID_dilutions += loading.get_CID_dilutions(kind,target_dilution=target_dilution)
    #print("Getting basic molecular data...")
    molecular_vectors = get_molecular_vectors(molecular_data,CID_dilutions)
    #print("Adding dilution data...")
    molecular_vectors = add_dilutions(molecular_vectors,CID_dilutions)
    #print("Building a matrix...")
    X = build_X(molecular_vectors,CID_dilutions)
    #print("Purging data with too many NaNs...")
    X,good1 = purge1_X(X,threshold=NAN_PURGE_THRESHOLD,good_molecular_descriptors=good1)
    #print("Imputing remaining NaN data...")
    X,imputer = impute_X(X)
    #print("Purging data that is still bad, if any...")
    X,good2 = purge2_X(X,good_molecular_descriptors=good2)
    #print("Normalizing data for fitting...")
    X,means,stds = normalize_X(X,means=means,stds=stds,target_dilution=target_dilution)
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

def add_dilutions(molecular_vectors,CID_dilutions,dilution=None):
    if 1:#dilution in [None,'low','high']:
        molecular_vectors_ = {}
        for CID_dilution in CID_dilutions:
            CID,dilution,high = [int(_) for _ in CID_dilution.split('_')]
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

# Build the X_obs matrix out of molecular descriptors.  
def build_X(molecular_vectors,CID_dilutions):
    X = np.vstack([molecular_vectors[key] for key in CID_dilutions])#sorted(molecular_vectors,key=lambda x:[int(_) for _ in x.split('_')])]) # Key could be CID or CID_dilution.  
    #print("The X matrix has shape (%dx%d) (molecules by molecular descriptors)" % X.shape)
    return X

def purge1_X(X,threshold=0.25,good_molecular_descriptors=None):
    if good_molecular_descriptors is None:
        good_molecular_descriptors = range(X.shape[1])
        # Which columns of X (which molecular descriptors) have NaNs for at least 'threshold' fraction of the molecules?  
        nan_molecular_descriptors = np.where(np.isnan(X).sum(axis=0) > X.shape[0]*threshold)[0] # At least 'threshold' NaNs.  
        # Purge these bad descriptors from the X matrix.  
        good_molecular_descriptors = list(set(range(X.shape[1])).difference(nan_molecular_descriptors))
    X = X[:,good_molecular_descriptors]
    #print("The X matrix has shape (%dx%d) (molecules by good molecular descriptors)" % X.shape)
    return X,good_molecular_descriptors

def impute_X(X):
    # The X_obs matrix (molecular descriptors) still has NaN values which need to be imputed.  
    imputer = Imputer(missing_values=np.nan,strategy='median',axis=0)
    X = imputer.fit_transform(X)
    #print("The X matrix now has shape (%dx%d) (molecules by non-NaN good molecular descriptors)" % X.shape)
    return X,imputer

def purge2_X(X,good_molecular_descriptors=None):
    if good_molecular_descriptors is None:
        good_molecular_descriptors = range(X.shape[1])
        zero_molecular_descriptors = np.where(np.abs(np.sum(X,axis=0))==0)[0] # All zeroes.  
        invariant_molecular_descriptors = np.where(np.std(X,axis=0)==0)[0] # Same value for all molecules.  
        bad_molecular_descriptors = set(zero_molecular_descriptors).union(invariant_molecular_descriptors) # Both of the above.  
        # Purge these bad descriptors from the X matrix.  
        good_molecular_descriptors = list(set(good_molecular_descriptors).difference(bad_molecular_descriptors))
    X = X[:,good_molecular_descriptors]
    #print("The X matrix has shape (%dx%d) (molecules by good molecular descriptors)" % X.shape)
    return X,good_molecular_descriptors

def normalize_X(X,means=None,stds=None,target_dilution=None):#,logs=None):
    num_cols = X.shape[1]
    if 1:#target_dilution in [None,'low','high']:
        num_cols -= 2
    X[:,:num_cols] = np.sign(X[:,:num_cols])*np.abs(X[:,:num_cols])**(1.0/3)
    if means is None:
        means = np.mean(X,axis=0)
    if stds is None:
        stds = np.std(X,axis=0)
    X[:,:num_cols] -= means[np.newaxis,:num_cols]
    X[:,:num_cols] /= stds[np.newaxis,:num_cols]
    return X,means,stds

#############
# Utilities #
#############

def purge(this,from_that):
    from_that = {CID:value for CID,value in from_that.items() if CID not in this}
    return from_that

def retain(this,in_that):
    in_that = {CID:value for CID,value in in_that.items() if CID in this}
    return in_that
