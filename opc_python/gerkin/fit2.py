import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.model_selection import ShuffleSplit,cross_val_score
from sklearn.linear_model import RandomizedLasso,Ridge

from opc_python import * # Import constants.  
from opc_python.utils import scoring,prog,ProgressBar,loading,DoubleSS,DreamGroupShuffleSplit
from opc_python.gerkin import dream

DESCRIPTORS = loading.get_descriptors(format=True)
N_DESCRIPTORS = len(DESCRIPTORS) 

def rfc_maker(n_estimators=100, max_features=100,
                  min_samples_leaf=None, max_depth=10,
                  seed=0, et=False):
    if not et: 
        kls = RandomForestRegressor
        kwargs = {'oob_score':False}
    else:
        kls = ExtraTreesRegressor
        kwargs = {}

    return kls(n_estimators=n_estimators, max_features=max_features,
                min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                n_jobs=-1, random_state=seed, **kwargs)

# Use random forest regression to fit the entire training data set, 
# one descriptor set at a time.  
def rfc_final(X, Y, Y_imp,
              max_features, min_samples_leaf, max_depth, et, use_mask,
              trans_params, trans_weight=0.75, X_test=None, Y_test=None,
              n_estimators=100, seed=0, quiet=False):
    
    if X_test is None:
        X_test = X
    if Y_test is None:
        Y_test = Y
        print(("No test data provided; "
               "score will indicate in-sample prediction quality"))
    x_index = X.index.values
    y_index = Y.index.values
    extras = set(x_index).difference(y_index)
    if len(extras):
        print("Removing %d CID/dilutions that are not in the observed data" \
              % len(extras))
        X = X.loc[y_index] # Make sure we are only fitting using the data we have.
    
    #Y_imp = {'mean':Y_imp.mean(axis=1,level=1),
    #         'std':Y_imp.std(axis=1,level=1)}
    #Y_mask = {'mean':Y_mask.mean(axis=1),
    #          'std':Y_mask.std(axis=1)}
    
       
    models = rfc_fit_models()

    
    observed = dream.filter_Y_dilutions(Y_test,'gold')
    score = scoring.score2(predicted,observed,quiet=quiet)
    return (rfcs,score)

def rfc_fit_models(X, Y, Y_imp, hp, n_estimators=100, seed=0, std=False):
    p = ProgressBar(N_DESCRIPTORS * (1+int(std)))
    models = {'mean':{}}
    if std:
        models['std'] = {}
    for d, descriptor in enumerate(DESCRIPTORS * (1+int(std))):
        kind = 'std' if d >= N_DESCRIPTORS else 'mean'
        p.animate(d, "Fitting %s %s" % (descriptor, kind))
        hp_d = hp.loc[descriptor]
        models[kind][descriptor] = rfc_maker(n_estimators=n_estimators,
                                             max_features=hp_d['max_features'],
                                             min_samples_leaf=hp_d['min_samples_leaf'],
                                             max_depth=hp_d['max_depth'],
                                             et=hp_d['use_et'],
                                             seed=seed)
        y = Y if hp_d['use_mask'] else Y_imp
        y = y.mean(axis=1, level='Descriptor') if kind=='mean' else \
            y.std(axis=1, level='Descriptor')
        y = y[descriptor]
        is_nan = np.isnan(y)
        y = y.loc[is_nan == False]
        x = X.loc[is_nan == False]
        models[kind][descriptor].fit(x,y)
    p.animate(None, "All descriptors' models have been fit")
    return models

def rfc_get_predictions(models, X, trans_params=None, dilution='gold'):
    X_d = dream.filter_X_dilutions(X, dilution)
    CIDs = list(X_d.index) # May be fewer for intensity than for others   
    predicted = {key: pd.DataFrame(index=CIDs, columns=DESCRIPTORS) for key in models}
    
    for kind in models:
        for d in DESCRIPTORS:
            X_d = dream.filter_X_dilutions(X, dilution)
            CIDs = list(X_d.index) # May be fewer for intensity than for others
            predicted[kind].loc[CIDs, d] = models[kind][d].predict(X_d)
            
    if 'std' in models:
        def f_transform(x, k0, k1):
            return 100*(k0*(x/100)**(k1*0.5) - k0*(x/100)**(k1*2))

        for d in DESCRIPTORS:
            try: # If it is a list or array
                tw = trans_weight[d]
            except TypeError: # If it is just a single value
                tw = trans_weight
            try:
                k0, k1 = trans_params[d]
            except TypeError:
                k0, k1 = trans_params
            except:
                tw = 0
            p_m = predicted['mean'][d]
            p_s = predicted['std'][d]
            predicted['std'][d] = tw*f_transform(p_m,k0,k1) + (1-tw)*p_s
    
    return predicted
    #for kind in predicted:
    #     = predicted.stack('Descriptor')

def get_observed(Y, dilution='gold'):
    Y = dream.filter_Y_dilutions(Y, dilution)
    Y = Y.stack('Descriptor', dropna=False)
    observed = {'mean': Y.mean(axis=1).unstack('Descriptor'),
                'stdev': Y.std(axis=1).unstack('Descriptor')}
    return observed

def rfc_(X_train,Y_train,X_test_int,X_test_other,Y_test,
         max_features=1500,n_estimators=1000,max_depth=None,min_samples_leaf=1):
    print(max_features)
    def rfc_maker():
        return RandomForestRegressor(max_features=max_features,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     n_jobs=-1,
                                     oob_score=True,
                                     random_state=0)
        
    rfc = rfc_maker()
    rfc.fit(X_train,Y_train)
    scores = {}
    for phase,X,Y in [('train',X_train,Y_train),
                      ('test',(X_test_int,X_test_other),Y_test)]:
        if phase == 'train':
            predicted = rfc.oob_prediction_
        else:
            predicted = rfc.predict(X[1])
            predicted_int = rfc.predict(X[0])
            predicted[:,0] = predicted_int[:,0]
            predicted[:,21] = predicted_int[:,21]
        observed = Y
        score = scoring.score2(predicted,observed)
        r_int = scoring.r2('int','mean',predicted,observed)
        r_ple = scoring.r2('ple','mean',predicted,observed)
        r_dec = scoring.r2('dec','mean',predicted,observed)
        r_int_sig = scoring.r2('int','std',predicted,observed)
        r_ple_sig = scoring.r2('ple','std',predicted,observed)
        r_dec_sig = scoring.r2('dec','std',predicted,observed)
        print(("For subchallenge 2, %s phase, "
               "score = %.2f (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)"
               % (phase,score,r_int,r_ple,r_dec,r_int_sig,r_ple_sig,r_dec_sig)))
        scores[phase] = (score,r_int,r_ple,r_dec,r_int_sig,r_ple_sig,r_dec_sig)

    return rfc,scores['train'],scores['test']

# Show that random forest regression also works really well out of sample.  
def rfc_cv(X,Y_imp,Y_mask,Y_test=None,n_splits=10,n_estimators=100,
           max_features=1500,min_samples_leaf=1,max_depth=None,rfc=True):
    if Y_mask is None:
        use_Y_mask = False
        Y_mask = Y_imp
    else:
        use_Y_mask = True
    if Y_test is None:
        Y_test = Y_mask
    if rfc:
        rfc_imp = RandomForestRegressor(max_features=max_features,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                oob_score=False,n_jobs=-1,random_state=0)
        rfc_mask = RandomForestRegressor(max_features=max_features,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                oob_score=False,n_jobs=-1,random_state=0)
    else:
        rfc_imp = ExtraTreesRegressor(max_features=max_features,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                  oob_score=False,n_jobs=-1,random_state=0)
        rfc_mask = ExtraTreesRegressor(max_features=max_features,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                  oob_score=False,n_jobs=-1,random_state=0)
    test_size = 0.2
    shuffle_split = ShuffleSplit(n_splits,test_size=test_size,random_state=0)
    n_molecules = len(Y_imp)
    test_size *= n_molecules
    rs = {'int':{'mean':[],'std':[],'trans':[]},
          'ple':{'mean':[],'std':[]},
          'dec':{'mean':[],'std':[]}}
    scores = []
    for train_index,test_index in shuffle_split.split(range(n_molecules)):
        rfc_imp.fit(X[train_index],Y_imp[train_index])
        predicted_imp = rfc_imp.predict(X[test_index])
        if use_Y_mask:
            rfc_mask.fit(X[train_index],Y_mask[train_index])
            predicted_mask = rfc_mask.predict(X[test_index])
        else:
            predicted_mask = predicted_imp
        observed = Y_test[test_index]
        rs_ = {'int':{},'ple':{},'dec':{}}
        for kind1 in ['int','ple','dec']:
            for kind2 in ['mean','std']:
                if kind2 in rs[kind1]:
                    if '%s_%s' % (kind1,kind2) in ['int_mean','ple_mean',
                                                   'dec_mean']:
                        r_ = scoring.r2(kind1,kind2,predicted_imp,observed)
                    else:
                        r_ = scoring.r2(kind1,kind2,predicted_mask,observed)
                    rs_[kind1][kind2] = r_
                    rs[kind1][kind2].append(r_)
        score = scoring.rs2score2(rs_)
        scores.append(score)
        rs['int']['trans'].append(scoring.r2(None,None,
                                             f_int(predicted_imp[:,0]),
                                             observed[:,21]))
    for kind1 in ['int','ple','dec']:
        for kind2 in ['mean','std','trans']:
            if kind2 in rs[kind1]:
                mean = np.mean(rs[kind1][kind2])
                sem = np.std(rs[kind1][kind2])/np.sqrt(n_splits)
                rs[kind1][kind2] = {'mean':mean,
                                    'sem':sem}
    scores = {'mean':np.mean(scores),'sem':np.std(scores)/np.sqrt(n_splits)}
    #print("For subchallenge 2, using cross-validation with:")
    #print("\tat most %s features:" % max_features)
    #print("\tat least %s samples per leaf:" % min_samples_leaf)
    #print("\tat most %s depth:" % max_depth)
    #print("\tscore = %.2f+/- %.2f" % (scores['mean'],scores['sem']))
    for kind2 in ['mean','std','trans']:
        for kind1 in ['int','ple','dec']:
            if kind2 in rs[kind1]:
                pass#print("\t%s_%s = %.3f+/- %.3f" % (kind1,kind2,rs[kind1][kind2]['mean'],rs[kind1][kind2]['sem']))
        
    return scores,rs

def f_int(x, k0=0.718, k1=1.08):
    return 100*(k0*(x/100)**(k1*0.5) - k0*(x/100)**(k1*2))

def scan(X_train,Y_train,X_test_int,X_test_other,Y_test,max_features=None,
         n_estimators=100):
    rfcs_max_features = {}
    ns = np.logspace(1,3.48,15)
    scores_train = []
    scores_test = []
    for n in ns:
        rfc_max_features,score_train,score_test = \
            rfc_(X_train,Y_train['mean_std'],X_test_int,X_test_other,
                 Y_test['mean_std'],max_features=int(n),n_estimators=100)
        scores_train.append(score_train)
        scores_test.append(score_test)
        rfcs_max_features[n] = rfc_max_features
    rs = ['int_m','ple_m','dec_m','int_s','ple_s','dec_s']
    for i,ri in enumerate(rs):
        print(ri)
        print('maxf ',ns.round(2))
        print('train',np.array(scores_train)[:,i].round(3))
        print('test ',np.array(scores_test)[:,i].round(3))
   
    return rfc_max_features,scores_train,scores_test
    #for n,train,test in zip(ns,scores_train,scores_test):
    #    print("max_features = %d, train = %.2f, test = %.2f" % (int(n),train,test))
    #return rfcs_max_features


def mask_vs_impute(X):
    print(2)
    Y_median,imputer = dream.make_Y_obs(['training','leaderboard'],
                                        target_dilution=None,imputer='median')
    Y_mask,imputer = dream.make_Y_obs(['training','leaderboard'],
                                      target_dilution=None,imputer='mask')
    r2s_median = rfc_cv(X,Y_median['mean_std'],Y_test=Y_mask['mean_std'],
                        n_splits=20,max_features=1500,n_estimators=200,
                        min_samples_leaf=1,rfc=True)
    r2s_mask = rfc_cv(X,Y_mask['mean_std'],n_splits=20,max_features=1500,
                      n_estimators=200,min_samples_leaf=1,rfc=True)
    return (r2s_median,r2s_mask)


def compute_linear_feature_ranks(X,Y,n_resampling=10):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    X = X.drop(['mean_dilution'],1)
    
    # Matrix to store the score rankings.  
    lin_ranked = np.zeros((21,X.shape[1])).astype(int) 
    
    rl = RandomizedLasso(alpha=0.025,selection_threshold=0.025,
                         n_resampling=n_resampling,random_state=25,n_jobs=1)
    for col in range(21):
        print("Computing feature ranks for descriptor #%d" % col)
        observed = Y[:,col]
        rl.fit(X,observed)
        lin_ranked[col,:] = np.argsort(rl.all_scores_.ravel())[::-1]
    return lin_ranked


def compute_linear_feature_ranks_cv(X,Y,n_resampling=10,n_splits=25):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Matrix to store the score rankings.  
    Y = Y['Subject'].mean(axis=1).unstack('Descriptor') # Mean across subjects
    X = X.drop('mean_dilution',1) # Don't use mean dilution to avoid leak
    common_index = Y.index.intersection(X.index)
    X = X.loc[common_index] # Only use common CIDs and dilutions
    Y = Y.loc[common_index] # Only use common CIDs and dilutions
    cv = DreamGroupShuffleSplit(n_splits,test_size=0.17,random_state=0) 
    CID_index = X.index.get_level_values('CID')
    descriptors = loading.get_descriptors(format=True)
    rl = RandomizedLasso(alpha=0.025,selection_threshold=0.025,
                         n_resampling=n_resampling,random_state=25,n_jobs=1)
    n_descriptors = len(descriptors)
    n_features = X.shape[1]
    lin_ranked = np.zeros((n_splits,n_descriptors,n_features)).astype(int) 
    p = ProgressBar(n_descriptors*n_splits)
    for d,desc in enumerate(descriptors):
        # Produce the correct train and test indices.  
        for j,(train,test) in enumerate(cv.split(X,CID_index,desc)):
            p.animate(d*n_splits+j,"Computing feature ranks for descriptor #%d, split #%d" \
                  % (d,j))
            observed = Y.loc[train][desc]
            #print(X.loc[train].shape)
            #print(observed.shape)
            rl.fit(X.loc[train],observed)
            lin_ranked[j,d,:] = np.argsort(rl.all_scores_.ravel())[::-1]
    p.animate(None,'Finished') 
    return lin_ranked


def master_cv(X,Y,n_estimators=50,n_splits=25,model='rf',
              alpha=10.0,random_state=0,feature_list=slice(None)):
    rs = np.zeros((21,n_splits))
    n_molecules = int(X.shape[0]/2)
     # This random state *must* be zero. 
    shuffle_split = ShuffleSplit(n_splits,test_size=0.17,random_state=0) 
    
    for col in range(21):
        print(col)
        observed = Y[:,col]
        cv = utils.DoubleSS(shuffle_split, n_molecules, col, X[:,-1])
        for j,(train,test) in enumerate(cv):
            #print(col,j)
            if model == 'rf':
                if col==0:
                    est = ExtraTreesRegressor(n_estimators=n_estimators,
                                              max_features=max_features[col], 
                                              max_depth=max_depth[col], 
                                        min_samples_leaf=min_samples_leaf[col],
                                              n_jobs=8,random_state=0)     
                else:
                    est = RandomForestRegressor(n_estimators=n_estimators,
                                                max_features=max_features[col], 
                                                max_depth=max_depth[col], 
                                        min_samples_leaf=min_samples_leaf[col],
                                                oob_score=False,n_jobs=8,
                                                random_state=0)
            elif model == 'ridge':
                est = Ridge(alpha=alpha,random_state=random_state)
            features = feature_list[j,col,:]
            est.fit(X[train,:][:,features],observed[train])
            predicted = est.predict(X[test,:][:,features])
            rs[col,j] = np.corrcoef(predicted,observed[test])[1,0]

        mean = rs[col,:].mean()
        sem = rs[col,:].std()/np.sqrt(n_splits)
        print(('Desc. %d: %.3f' % (col,mean)))
    return rs


def feature_sweep(X,Y,n_estimators=50,n_splits=25,
                  n_features_list=[1,2,3,4,5,10,33,100,333,1000,3333,10000],
                  model='rf',rfe=False,wrong_split=False,max_features='auto',
                  max_depth=None,min_samples_leaf=1,alpha=1.0,
                  lin_ranked=None,random_state=0):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if model == 'ridge' and lin_ranked is None:
        raise Exception('Must provided "lin_ranked" to use the linear model')
    Y = Y['Subject'].mean(axis=1).unstack('Descriptor') # Mean across subjects
    X = X.drop('mean_dilution',1) # Don't use mean dilution to avoid leak
    X = X.loc[Y.index] # Only use CIDs and dilutions with perceptual data
    rs = np.ma.zeros((21,len(n_features_list),n_splits)) # Empty matrix to store correlations.  
    cv = DreamGroupShuffleSplit(n_splits,test_size=0.17,random_state=0) 
    CID_index = X.index.get_level_values('CID')
    descriptors = loading.get_descriptors(format=True)
    n_descriptors = len(descriptors)

    p = ProgressBar(n_descriptors*n_splits)
    for d,desc in enumerate(descriptors): # For each descriptor.  
        observed = Y[desc] # Perceptual data for this descriptor.  
        n_features_list_ = list(np.array(n_features_list)+(desc=='Intensity'))
        # Produce the correct train and test indices.  
        for j,(train,test) in enumerate(cv.split(X,CID_index,desc)):
            p.animate(d*n_splits+j,'Fitting descriptor #%d, split #%d' % (d,j))
            if model == 'rf': # If the model is random forest regression.  
                if desc=='Intensity':
                    est = ExtraTreesRegressor(n_estimators=n_estimators,
                                              max_features=max_features[d],
                                              max_depth=max_depth[d],
                                              min_samples_leaf=min_samples_leaf[d],
                                              n_jobs=8,
                                              random_state=random_state)
                else:
                    est = RandomForestRegressor(n_estimators=n_estimators,
                                                max_features=max_features[d],
                                                max_depth=max_depth[d],
                                            min_samples_leaf=min_samples_leaf[d],
                                                oob_score=False,n_jobs=8,
                                                random_state=random_state)
            elif model == 'ridge': # If the model is ridge regression. 
                est = Ridge(alpha=alpha,fit_intercept=True,normalize=False, 
                            copy_X=True,max_iter=None,tol=0.001,solver='auto',
                            random_state=random_state)
            if rfe:  
                rfe = RFE(estimator=est, step=n_features_list_, 
                          n_features_to_select=1)
                rfe.fit(X.loc[train],observed.loc[train])    
            else:  
                # Fit the model on the training data.  
                est.fit(X.loc[train].values,observed.loc[train].values) 
                if model == 'rf':
                    # Use feature importances to get ranks.
                    import_ranks = np.argsort(est.feature_importances_)[::-1]   
                elif model == 'ridge':
                    # Use the pre-computed ranks.
                    import_ranks = lin_ranked[int(j+wrong_split)%n_splits,d,:] 
            for i,n_feat in enumerate(n_features_list_):
                if desc=='Intensity':
                    # Add one for intensity since negLogD is worthless when
                    # all concentrations are 1/1000. 
                    n_feat += 1  
                if hasattr(est,'max_features') \
                and est.max_features not in [None,'auto']:
                    if n_feat < est.max_features:
                        est.max_features = n_feat
                if rfe:
                    est.fit(X.loc[train].ix[:,rfe.ranking_<=(1+i)],observed.loc[train])
                    predicted = est.predict(X.loc[test].ix[:,rfe.ranking_<=(1+i)])
                else:
                    #est.max_features = None
                    # Fit the model on the training data
                    # with 'max_features' features.
                    est.fit(X.loc[train].ix[:,import_ranks[:n_feat]],observed.loc[train])
                    # Predict the test data.  
                    predicted = est.predict(X.loc[test].ix[:,import_ranks[:n_feat]]) 
                # Compute the correlation coefficient.
                rs[d,i,j] = np.corrcoef(predicted,observed.loc[test])[1,0] 
    p.animate(None,'Finished') 
    return rs
