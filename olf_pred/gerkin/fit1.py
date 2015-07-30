import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.cross_validation import ShuffleSplit,cross_val_score
from sklearn.linear_model import Lasso

from olf_pred.utils import scoring

def rfc_final(X,Y,
              max_features,min_samples_leaf,max_depth,et,
              Y_test=None,regularize=[0.7,0.7,0.7],n_estimators=100,seed=0):
    
    if Y_test is None:
        Y_test = Y
    
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
        
    kinds = ['int','ple','dec']
    rfcs = {}
    for kind in kinds:
        rfcs[kind] = {} 
        for subject in range(1,50):
            rfcs[kind][subject] = rfc_maker(n_estimators=n_estimators,
                                max_features=max_features[kind],
                                min_samples_leaf=min_samples_leaf[kind],
                                max_depth=max_depth[kind],
                                et=et[kind])

    for subject in range(1,50):
        for kind in kinds:
            rfcs[kind][subject].fit(X,Y[subject])
    
    predictions = {}
    for kind in kinds:
        predictions[kind] = {}
        for subject in range(1,50):
            if et[kind]:
                # Check in-sample fit because there isn't any alternative. 
                predictions[kind][subject] = rfcs[kind][subject].predict(X)
            else:
                predictions[kind][subject] = rfcs[kind][subject].oob_prediction_

    predicted = predictions['int'].copy()
    for subject in range(1,50):
        predicted[subject][:,0] = predictions['int'][subject][:,0]
        predicted[subject][:,1] = predictions['ple'][subject][:,1]
        predicted[subject][:,2:] = predictions['dec'][subject][:,2:]

    # Regularize:  
    predicted_stack = np.zeros((predicted[1].shape[0],predicted[1].shape[1],49))
    for subject in range(1,50):
        predicted_stack[:,:,subject-1] = predicted[subject]
    predicted_mean = predicted_stack.mean(axis=2,keepdims=True)
    predicted_reg = {kind:predicted.copy() for kind in kinds}
    for i,kind in enumerate(kinds):
        predicted_reg[kind] = regularize[i]*predicted_mean + (1-regularize[i])*predicted_stack
    predicted_stack[:,0,:] = predicted_reg['int'][:,0,:]
    predicted_stack[:,1,:] = predicted_reg['ple'][:,1,:]
    predicted_stack[:,2:,:] = predicted_reg['dec'][:,2:,:]
    predicted = predicted_stack

    observed = predicted.copy()
    for subject in range(1,50):
        observed[:,:,subject-1] = Y_test[subject]
    score = scoring.score(predicted,observed)
    rs = {}
    predictions = {}
    for kind in ['int','ple','dec']:
        rs[kind] = scoring.r(kind,predicted,observed)
    
    print("For subchallenge 1:")
    print("\tScore = %.2f" % score)
    for kind in kinds:
        print("\t%s = %.3f" % (kind,rs[kind]))
    
    return (rfcs,score,rs)

def rfc_(X_train,Y_train,X_test_int,X_test_other,Y_test,max_features=1500,n_estimators=1000,max_depth=None,min_samples_leaf=1):
    print(max_features)
    def rfc_maker():
        return RandomForestRegressor(max_features=max_features,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     n_jobs=-1,
                                     oob_score=True,
                                     random_state=0)
    n_subjects = 49
    predicted_train = []
    observed_train = []
    predicted_test = []
    observed_test = []
    rfcs = {subject:rfc_maker() for subject in range(1,n_subjects+1)}
    for subject in range(1,n_subjects+1):
        print(subject)
        observed = Y_train[subject]
        rfc = rfcs[subject]
        rfc.fit(X_train,observed)
        #predicted = rfc.predict(X_train)
        predicted = rfc.oob_prediction_
        observed_train.append(observed)
        predicted_train.append(predicted)

        observed = Y_test[subject]
        rfc = rfcs[subject]
        if Y_train is Y_test: # OOB prediction  
            predicted = rfc.oob_prediction_
        else:
            predicted = rfc.predict(X_test_other)
            predicted_int = rfc.predict(X_test_int)
            predicted[:,0] = predicted_int[:,0]
        observed_test.append(observed)
        predicted_test.append(predicted)
    scores = {}
    for phase,predicted_,observed_ in [('train',predicted_train,observed_train),('test',predicted_test,observed_test)]:
        predicted = np.dstack(predicted_)
        observed = np.ma.dstack(observed_)
        predicted_mean = np.mean(predicted,axis=2,keepdims=True)
        regularize = 0.7
        predicted = regularize*(predicted_mean) + (1-regularize)*predicted
        score = scoring.score(predicted,observed,n_subjects=n_subjects)
        r_int = scoring.r('int',predicted,observed)
        r_ple = scoring.r('ple',predicted,observed)
        r_dec = scoring.r('dec',predicted,observed)
        print("For subchallenge 1, %s phase, score = %.2f (%.2f,%.2f,%.2f)" % (phase,score,r_int,r_ple,r_dec))
        scores[phase] = score
    return rfcs,scores['train'],scores['test']

# Show that random forest regrssion also works really well out of sample.  
def rfc_cv(X,Y,n_splits=5,n_estimators=15,
           max_features=1000,min_samples_leaf=1,max_depth=None,regularize=[0.7,0.35,0.7]):
    test_size = 0.2
    n_molecules = X.shape[0]
    shuffle_split = ShuffleSplit(n_molecules,n_splits,test_size=test_size)
    test_size *= n_molecules
    rfcs = {}
    n_subjects = 49
    for subject in range(1,n_subjects+1):
        rfc = RandomForestRegressor(n_estimators=n_estimators,
                                    max_features=max_features,
                                    min_samples_leaf=min_samples_leaf,
                                    max_depth=max_depth,
                                    oob_score=False,
                                    n_jobs=-1,
                                    random_state=0)
        rfcs[subject] = rfc
    rs = {'int':[],
          'ple':[],
          'dec':[]}
    scores = []
    for train_index,test_index in shuffle_split:
        predicted_list = []
        observed_list = []
        for subject in range(1,n_subjects+1):
            rfc = rfcs[subject]
            X_train = X[train_index]
            Y_train = Y[subject][train_index]
            rfc.fit(X_train,Y_train)
            X_test = X[test_index]
            predicted = rfc.predict(X_test)
            observed = Y[subject][test_index]
            predicted_list.append(predicted)
            observed_list.append(observed)
        observed = np.ma.dstack(observed_list)
        predicted = np.dstack(predicted_list)
        predicted_mean = predicted.mean(axis=2,keepdims=True)
        predicted_int = regularize[0]*(predicted_mean) + (1-regularize[0])*predicted
        predicted_ple = regularize[1]*(predicted_mean) + (1-regularize[1])*predicted
        predicted = regularize[2]*(predicted_mean) + (1-regularize[2])*predicted
        predicted[:,0,:] = predicted_int[:,0,:]
        predicted[:,1,:] = predicted_ple[:,1,:]
        score = scoring.score(predicted,observed)
        scores.append(score)
        for kind in ['int','ple','dec']:
            rs[kind].append(scoring.r(kind,predicted,observed))
    for kind in ['int','ple','dec']:
        rs[kind] = {'mean':np.mean(rs[kind]),'sem':np.std(rs[kind])/np.sqrt(n_splits)}
    scores = {'mean':np.mean(scores),'sem':np.std(scores)/np.sqrt(n_splits)}
    print("For subchallenge 1, using cross-validation with at least %d samples_per_leaf:" % min_samples_leaf)
    print("\tscore = %.2f+/- %.2f" % (scores['mean'],scores['sem']))
    for kind in ['int','ple','dec']:
        print("\t%s = %.2f+/- %.2f" % (kind,rs[kind]['mean'],rs[kind]['sem']))
            
    return scores,rs

# Using only subject fits.  
def subject_regularize(rfcs,X_int,X_other,Y,oob=False,regularize=[0.75,0.3,0.65]):
    if len(regularize)==1:
        regularize = regularize*3
    observed_ = []
    predicted_ = []
    for subject in range(1,50):
        observed = Y['subject'][subject]
        rfc = rfcs[1][subject]
        if oob:
            predicted = rfc.oob_prediction_
        else:
            predicted = rfc.predict(X_other)
            predicted_int = rfc.predict(X_int)
            predicted[:,0] = predicted_int[:,0]
        observed_.append(observed)
        predicted_.append(predicted)
    predicted = np.dstack(predicted_)
    observed = np.ma.dstack(observed_)
    predicted_mean = np.mean(predicted,axis=2,keepdims=True)
    predicted_std = np.std(predicted,axis=2,keepdims=True)
    predicted_mean_std = np.hstack((predicted_mean,predicted_std)).squeeze()
    predicted_int = regularize[0]*(predicted_mean) + (1-regularize[0])*predicted
    predicted_ple = regularize[1]*(predicted_mean) + (1-regularize[1])*predicted
    predicted_dec = regularize[2]*(predicted_mean) + (1-regularize[2])*predicted
    predicted = regularize[0]*(predicted_mean) + (1-regularize[0])*predicted
    r_int = scoring.r('int',predicted_int,observed)
    r_ple = scoring.r('ple',predicted_ple,observed)
    r_dec = scoring.r('dec',predicted_dec,observed)
    score1_ = scoring.score(predicted,observed,n_subjects=49)
    score1 = scoring.rs2score(r_int,r_ple,r_dec)
    #print(score1_,score1)
    print("For subchallenge %d, score = %.3f (%.3f,%.3f,%.3f)" % (1,score1,r_int,r_ple,r_dec))
    score2 = scoring.score2(predicted_mean_std,Y['mean_std'])
    r_int_mean = scoring.r2('int','mean',predicted_mean_std,Y['mean_std'])
    r_ple_mean = scoring.r2('ple','mean',predicted_mean_std,Y['mean_std'])
    r_dec_mean = scoring.r2('dec','mean',predicted_mean_std,Y['mean_std'])
    r_int_sigma = scoring.r2('int','sigma',predicted_mean_std,Y['mean_std'])
    r_ple_sigma = scoring.r2('ple','sigma',predicted_mean_std,Y['mean_std'])
    r_dec_sigma = scoring.r2('dec','sigma',predicted_mean_std,Y['mean_std'])
    print("For subchallenge %d, score = %.2f (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)" % \
         (2,score2,r_int_mean,r_ple_mean,r_dec_mean,r_int_sigma,r_ple_sigma,r_dec_sigma))
    return (r_int,r_ple,r_dec,r_int_mean,r_ple_mean,r_dec_mean,r_int_sigma,r_ple_sigma,r_dec_sigma)

def lasso_(X_train,Y_train,X_test,Y_test,alpha=0.1,regularize=[0.7,0.7,0.7]):
    if len(regularize)==1:
        regularize = regularize*3
    def lasso_maker():
        return Lasso(alpha=alpha)
    n_subjects = 49
    predicted_train = []
    observed_train = []
    predicted_test = []
    observed_test = []
    lassos = {subject:lasso_maker() for subject in range(1,n_subjects+1)}
    for subject in range(1,n_subjects+1):
        observed = Y_train[subject][:,1:2]
        lasso = lassos[subject]
        lasso.fit(X_train,observed)
        predicted = lasso.predict(X_train)[:,np.newaxis]
        observed_train.append(observed)
        predicted_train.append(predicted)

        observed = Y_test[subject][:,1:2]
        predicted = lasso.predict(X_test)[:,np.newaxis]
        observed_test.append(observed)
        predicted_test.append(predicted)
    scores = {}
    for phase,predicted_,observed_ in [('train',predicted_train,observed_train),('test',predicted_test,observed_test)]:
        predicted = np.dstack(predicted_)
        observed = np.ma.dstack(observed_)
        predicted_mean = np.mean(predicted,axis=2,keepdims=True)
        #predicted_int = regularize[0]*(predicted_mean) + (1-regularize[0])*predicted
        predicted_ple = regularize[1]*(predicted_mean) + (1-regularize[1])*predicted
        #predicted_dec = regularize[2]*(predicted_mean) + (1-regularize[2])*predicted
        #score1_ = scoring.score(predicted_int,observed,n_subjects=n_subjects)
        #r_int = scoring.r('int',predicted,observed)
        #r_ple = scoring.r('ple',predicted,observed)
        r_ple = scoring.r(None,predicted_ple,observed)
        r2_ple = scoring.r2(None,None,predicted_ple.mean(axis=2),observed.mean(axis=2))
        #r_dec = scoring.r('dec',predicted,observed)
        #score1 = scoring.rs2score(r_int,r_ple,r_dec)
        print("For subchallenge 1, %s phase, score = %.2f" % (phase,r_ple))
        print("For subchallenge 2, %s phase, score = %.2f" % (phase,r2_ple))
        scores[phase] = (r_ple,r2_ple)
    return lassos,scores['train'],scores['test']

