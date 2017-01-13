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

def get_perceptual_matrices(perceptual_data,target_dilution=None,
                            use_replicates=True,subjects=range(1,50)):
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
        if subject in subjects:
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

def make_X(molecular_data,kinds,CID_dilutions=None,target_dilution=None,threshold=None,bad=None,
           good1=None,good2=None,means=None,stds=None,raw=False,quiet=False):
    if type(kinds) is str:
        kinds = [kinds]
    if threshold is None:
        threshold = NAN_PURGE_THRESHOLD
    #print("Getting CIDs and dilutions...")
    if CID_dilutions is None:
        CID_dilutions = []
        for kind in kinds:
            assert kind in KINDS, "No such kind %s" % kind
            CID_dilutions += loading.get_CID_dilutions(kind,target_dilution=target_dilution)
    #print("Getting basic molecular data...")
    molecular_vectors = get_molecular_vectors(molecular_data,CID_dilutions)
    #print("Adding dilution data...")
    molecular_vectors = add_dilutions(molecular_vectors,CID_dilutions)
    #print("Building a matrix...")
    X = build_X(molecular_vectors,CID_dilutions)
    if not raw:
        if bad:
            good0 = list(set(range(X.shape[1])).difference(bad))
            X = X[:,good0]
        #print("Purging data with too many NaNs...")
        X,good1 = purge1_X(X,threshold=NAN_PURGE_THRESHOLD,good_molecular_descriptors=good1)
        #print("Imputing remaining NaN data...")
        X,imputer = impute_X(X)
        #print("Purging data that is still bad, if any...")
        X,good2 = purge2_X(X,good_molecular_descriptors=good2)
        #print("Normalizing data for fitting...")
        X,means,stds = normalize_X(X,means=means,stds=stds,target_dilution=target_dilution)
    else:
        good1,good2 = list(range(X.shape[1])),list(range(X.shape[1]))
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

def add_dilutions(molecular_vectors,CID_dilutions,dilution=None):
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
        stds[stds==0] = 1
    X[:,:num_cols] -= means[np.newaxis,:num_cols]
    X[:,:num_cols] /= stds[np.newaxis,:num_cols]
    return X,means,stds

def get_molecular_data(sources,CIDs):
    import pandas
    if 1 or ('dragon' in sources):
        molecular_headers, molecular_data = loading.load_molecular_data(CIDs=CIDs)
        print("Dragon has %d features for %d molecules." % \
                (len(molecular_headers)-1,len(molecular_data)))
    if 'episuite' in sources:
        x = pd.read_table('%s/DREAM_episuite_descriptors.txt' % DATA,index_col=0).drop('SMILES',1)
        x = x.loc[CIDs]
        x.iloc[:,47] = 1*(x.iloc[:,47]=='YES ')
        episuite = x.as_matrix()
        print("Episuite has %d features for %d molecules." % (episuite.shape[1],episuite.shape[0]))
    '''
    if 'verbal' in sources:
        verbal = pandas.read_table('%s/name_features.txt' % DATA, sep='\t', header=None)
        verbal = verbal.as_matrix()[:,1:]
        print("Verbal has %d features for %d molecules." % (verbal.shape[1],verbal.shape[0]))
    '''
    if 'morgan' in sources:
        morgan = pd.read_csv('%s/morgan_sim.csv' % DATA, index_col=0)
        morgan.index.rename('CID',inplace=True)
        morgan = morgan.loc[CIDs].as_matrix()
        print("Morgan has %d features for %d molecules." % (morgan.shape[1],morgan.shape[0]))
    if 'nspdk' in sources:
        nspdk_dict = make_nspdk_dict(CIDs)
        nspdk = np.zeros((len(CIDs),len(nspdk_dict)))
        for j,(feature,facts) in enumerate(nspdk_dict.items()):
            for CID,value in facts.items():
                i = CIDs.index(CID)
                nspdk[i,j] = value
        print("NSPDK has %d features for %d molecules." % (nspdk.shape[1],nspdk.shape[0]))
    if 'gramian' in sources:
        nspdk_CIDs = pd.read_csv('%s/derived/nspdk_cid.csv' % DATA, 
                                 header=None, dtype='int').as_matrix().squeeze()
        # These require a large file that is not on GitHub, but can be obtained separately.  
        nspdk_gramian = pandas.read_table('%s/derived/nspdk_r3_d4_unaug_gramian.mtx' % DATA, delimiter=' ', header=None)
        nspdk_gramian = nspdk_gramian.as_matrix()
        CID_indices = [list(nspdk_CIDs).index(CID) for CID in CIDs]
        nspdk_gramian = nspdk_gramian[CID_indices,:]
        print("NSPDK Gramian has %d features for %d molecules." % \
              (nspdk_gramian.shape[1],nspdk_gramian.shape[0]))

    # Add all these new features to the molecular data dict.  
    mdx = []
    for i,line in enumerate(molecular_data):
        CID = int(line[0])
        if CID in CIDs:
            index = CIDs.index(CID)
            if 'episuite' in sources:
                line += list(episuite[index])
            if 'morgan' in sources:
                line += list(morgan[index])
            if 'nspdk' in sources:
                line += list(nspdk[index])
            if 'gramian' in sources:
                line += list(nspdk_gramian[index])
            mdx.append(line)
    print("There are now %d total features." % (len(mdx[0])-1))
    return molecular_data

def make_nspdk_dict(CIDs):
    nspdk_CIDs = pd.read_csv('%s/derived/nspdk_cid.csv' % DATA, 
                                 header=None, dtype='int').as_matrix().squeeze()
    # Start to load the NSPDK features.  
    with open('%s/derived/nspdk_r3_d4_unaug.svm' % DATA) as f:
        nspdk_dict = {}
        i = 0
        while True:
            x = f.readline()
            if not len(x):
                break
            CID = nspdk_CIDs[i]
            i += 1
            if CID in CIDs:
                key_vals = x.split(' ')[1:]
                for key_val in key_vals:
                    key,val = key_val.split(':')
                    if key in nspdk_dict:
                        nspdk_dict[key][CID] = val
                    else:
                        nspdk_dict[key] = {CID:val}
    # Only include NSPDK features known for more than one of our CIDs
    nspdk_dict = {key:value for key,value in nspdk_dict.items() if len(value)>1} 
    return nspdk_dict

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
