import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.cross_validation import ShuffleSplit,cross_val_score

from olf_pred import * # Import constants.  
from olf_pred.utils import scoring
from olf_pred.gerkin import dream

# Use random forest regression to fit the entire training data set, one descriptor set at a time.  
def rfc_final(X,Y_imp,Y_mask,
              max_features,min_samples_leaf,max_depth,et,use_mask,
              Y_test=None,n_estimators=100,seed=0):
    
    if Y_test is None:
        Y_test = Y_mask
    def rfc_maker(n_estimators=n_estimators,max_features=max_features,
                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,et=False):
        if not et: 
            return RandomForestRegressor(n_estimators=n_estimators,
                                     max_features=max_features,
                                     min_samples_leaf=min_samples_leaf,
                                     max_depth=max_depth,
                                     oob_score=True,
                                     n_jobs=-1,random_state=seed)
        else:
            return ExtraTreesRegressor(n_estimators=n_estimators,
                                max_features=max_features,
                                min_samples_leaf=min_samples_leaf,
                                max_depth=max_depth,
                                n_jobs=-1,random_state=seed)
        
    rfcs = {}
    for kind in ['int','ple','dec']:
        rfcs[kind] = {} 
        for moment in ['mean','sigma']:
            rfcs[kind][moment] = rfc_maker(n_estimators=n_estimators,
                                max_features=max_features[kind][moment],
                                min_samples_leaf=min_samples_leaf[kind][moment],
                                max_depth=max_depth[kind][moment],
                                et=et[kind][moment])

    for kind in ['int','ple','dec']:
        for moment in ['mean','sigma']:
            if use_mask[kind][moment]:
                rfcs[kind][moment].fit(X,Y_mask)
            else:
                rfcs[kind][moment].fit(X,Y_imp)
    
    predictions = {}
    for kind in ['int','ple','dec']:
        predictions[kind] = {}
        for moment in ['mean','sigma']:
            if et[kind][moment]:
                # Check in-sample fit because there isn't any alternative.  
                predictions[kind][moment] = rfcs[kind][moment].predict(X)
            else:
                predictions[kind][moment] = rfcs[kind][moment].oob_prediction_
    predicted = predictions['int']['mean'].copy()
    for i,moment in enumerate(['mean','sigma']):
        predicted[:,(0+21*i)] = predictions['int'][moment][:,(0+21*i)]
        predicted[:,(1+21*i)] = predictions['ple'][moment][:,(1+21*i)]
        predicted[:,(2+21*i):(21+21*i)] = predictions['dec'][moment][:,(2+21*i):(21+21*i)]

    observed = Y_test
    score = scoring.score2(predicted,observed)
    rs = {}
    predictions = {}
    for kind in ['int','ple','dec']:
        rs[kind] = {}
        for moment in ['mean','sigma']:
            rs[kind][moment] = scoring.r2(kind,moment,predicted,observed)
    rs['int']['trans'] = scoring.r2(None,None,f_int(predicted[:,0]),observed[:,0])

    print("For subchallenge 2:")
    print("\tScore = %.2f" % score)
    for kind in ['int','ple','dec']:
        for moment in ['mean','sigma']: 
            print("\t%s_%s = %.3f" % (kind,moment,rs[kind][moment]))
    
    return (rfcs,score,rs)

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
    for phase,X,Y in [('train',X_train,Y_train),('test',(X_test_int,X_test_other),Y_test)]:
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
        r_int_sig = scoring.r2('int','sigma',predicted,observed)
        r_ple_sig = scoring.r2('ple','sigma',predicted,observed)
        r_dec_sig = scoring.r2('dec','sigma',predicted,observed)
        print("For subchallenge 2, %s phase, score = %.2f (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)" \
                % (phase,score,r_int,r_ple,r_dec,r_int_sig,r_ple_sig,r_dec_sig))
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
    shuffle_split = ShuffleSplit(len(Y_imp),n_splits,test_size=test_size)
    test_size *= len(Y_imp)
    rs = {'int':{'mean':[],'sigma':[],'trans':[]},'ple':{'mean':[],'sigma':[]},'dec':{'mean':[],'sigma':[]}}
    scores = []
    for train_index,test_index in shuffle_split:
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
            for kind2 in ['mean','sigma']:
                if kind2 in rs[kind1]:
                    if '%s_%s' % (kind1,kind2) in ['int_mean','ple_mean','dec_mean']:
                        r_ = scoring.r2(kind1,kind2,predicted_imp,observed)
                    else:
                        r_ = scoring.r2(kind1,kind2,predicted_mask,observed)
                    rs_[kind1][kind2] = r_
                    rs[kind1][kind2].append(r_)
        score = scoring.rs2score2(rs_)
        scores.append(score)
        rs['int']['trans'].append(scoring.r2(None,None,f_int(predicted_imp[:,0]),observed[:,21]))
    for kind1 in ['int','ple','dec']:
        for kind2 in ['mean','sigma','trans']:
            if kind2 in rs[kind1]:
                rs[kind1][kind2] = {'mean':np.mean(rs[kind1][kind2]),'sem':np.std(rs[kind1][kind2])/np.sqrt(n_splits)}
    scores = {'mean':np.mean(scores),'sem':np.std(scores)/np.sqrt(n_splits)}
    print("For subchallenge 2, using cross-validation with:")
    print("\tat most %s features:" % max_features)
    print("\tat least %s samples per leaf:" % min_samples_leaf)
    print("\tat most %s depth:" % max_depth)
    print("\tscore = %.2f+/- %.2f" % (scores['mean'],scores['sem']))
    for kind2 in ['mean','sigma','trans']:
        for kind1 in ['int','ple','dec']:
            if kind2 in rs[kind1]:
                print("\t%s_%s = %.3f+/- %.3f" % (kind1,kind2,rs[kind1][kind2]['mean'],rs[kind1][kind2]['sem']))
        
    return scores,rs

def f_int(x, k0=0.718, k1=1.08):
    return 100*(k0*(x/100)**(k1*0.5) - k0*(x/100)**(k1*2))

def scan(X_train,Y_train,X_test_int,X_test_other,Y_test,max_features=None,n_estimators=100):
    rfcs_max_features = {}
    ns = np.logspace(1,3.48,15)
    scores_train = []
    scores_test = []
    for n in ns:
        rfc_max_features,score_train,score_test = rfc_(X_train,Y_train['mean_std'],
                                          X_test_int,X_test_other,
                                          Y_test['mean_std'],
                                          max_features=int(n),n_estimators=100)
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
    Y_median,imputer = dream.make_Y_obs(['training','leaderboard'],target_dilution=None,imputer='median')
    Y_mask,imputer = dream.make_Y_obs(['training','leaderboard'],target_dilution=None,imputer='mask')
    r2s_median = rfc_cv(X,Y_median['mean_std'],Y_test=Y_mask['mean_std'],n_splits=20,max_features=1500,n_estimators=200,min_samples_leaf=1,rfc=True)
    r2s_mask = rfc_cv(X,Y_mask['mean_std'],n_splits=20,max_features=1500,n_estimators=200,min_samples_leaf=1,rfc=True)
    return (r2s_median,r2s_mask)