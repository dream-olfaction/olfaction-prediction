import numpy as np
from scipy.stats import pearsonr

from __init__ import *
SIGMAS = {'int': 0.0187,
          'ple': 0.0176,
          'dec': 0.0042}

SIGMAS2 = {'int_mean': 0.1193,
           'ple_mean': 0.1265,
           'dec_mean': 0.0265,
           'int_sigma': 0.1194,
           'ple_sigma': 0.1149,
           'dec_sigma': 0.0281}

# Scoring for sub-challenge 1.
def r(kind,predicted,observed,n_subjects=49,mask=True):
    # Predicted and observed should each be an array of 
    # molecules by perceptual descriptors by subjects
    
    r = 0.0
    for subject in range(n_subjects):
        p = predicted[:,:,subject]
        o = observed[:,:,subject]
        r += r2(kind,None,p,o,mask=mask)
    r /= n_subjects
    return r

def score(predicted,observed,n_subjects=49):
    """Final score for sub-challenge 1."""
    score = z('int',predicted,observed,n_subjects=n_subjects) \
          + z('ple',predicted,observed,n_subjects=n_subjects) \
          + z('dec',predicted,observed,n_subjects=n_subjects)
    return score/3.0

def score_summary(predicted,observed,mask=True):
    score_ = score(predicted,observed)
    r_int = r('int',predicted,observed,mask=mask)
    r_ple = r('ple',predicted,observed,mask=mask)
    r_dec = r('dec',predicted,observed,mask=mask)
    return 'Score: %3f; rs = %.3f,%.3f,%.3f' % \
                (score_, r_int, r_ple, r_dec)

def rs2score(r_int,r_ple,r_dec):
    z_int = r_int/SIGMAS['int']
    z_ple = r_ple/SIGMAS['ple']
    z_dec = r_dec/SIGMAS['dec']
    return (z_int+z_ple+z_dec)/3.0

def z(kind,predicted,observed,n_subjects=49): 
    sigma = SIGMAS[kind]
    shuffled_r = 0#r2(kind,predicted,shuffled)
    observed_r = r(kind,predicted,observed,n_subjects=n_subjects)
    return (observed_r - shuffled_r)/sigma

# Scoring for sub-challenge 2.  

def r2(kind,moment,predicted,observed,mask=False):
    # Predicted and observed should each be an array of 
    # molecules by 2*perceptual descriptors (means then stds)
    if moment == 'mean':
        p = predicted[:,:21]
        o = observed[:,:21]
    elif moment == 'sigma':
        p = predicted[:,21:]
        o = observed[:,21:]
    elif moment is None:
        p = predicted
        o = observed
    else:
        raise ValueError('No such moment: %s' % moment)
    
    if kind=='int':
        p = p[:,0]
        o = o[:,0]
    elif kind=='ple':
        p = p[:,1]
        o = o[:,1]
    elif kind == 'dec':
        p = p[:,2:]
        o = o[:,2:]
    elif kind in range(19):
        p = p[:,2+kind]
        o = o[:,2+kind]
    elif kind is None:
        p = p
        o = o
    else:
        raise ValueError('No such kind: %s' % kind)
    
    if len(p.shape)==1:
        p = p.reshape(-1,1)
    if len(o.shape)==1:
        o = o.reshape(-1,1)
    r = 0.0
    cols = p.shape[1]
    denom = 0.0
    for i in range(cols):
        p_ = p[:,i]
        o_ = o[:,i]
        if mask:
            r_ = np.ma.corrcoef(p_,o_)[0,1]
            if ('%f' % r_) == 'nan':
                denom += 1 # Done to match DREAM scoring.  
                continue
        else:
            r_ = pearsonr(p_,o_)[0]
            if np.isnan(r_):
                denom += 1 # Done to match DREAM scoring.  
                print('NaN')
                if np.std(p_)*np.std(o_) != 0:
                    print('WTF')
                continue
        r += r_
        denom += 1
    if denom == 0.0:
        r = np.nan
    else:
        r /= denom
    return r

def score2(predicted,observed):
    """Final score for sub-challenge 2."""
    score = z2('int','mean',predicted,observed) \
          + z2('ple','mean',predicted,observed) \
          + z2('dec','mean',predicted,observed) \
          + z2('int','sigma',predicted,observed) \
          + z2('ple','sigma',predicted,observed) \
          + z2('dec','sigma',predicted,observed)
    return score/6.0

def score_summary2(predicted,observed,mask=False):
    score = score2(predicted,observed)
    r_int_mean = r2('int','mean',predicted,observed)
    r_ple_mean = r2('ple','mean',predicted,observed)
    r_dec_mean = r2('dec','mean',predicted,observed)
    r_int_sigma = r2('int','sigma',predicted,observed)
    r_ple_sigma = r2('ple','sigma',predicted,observed)
    r_dec_sigma = r2('dec','sigma',predicted,observed)
    return 'Score: %3f; rs = %.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % \
                (score, r_int_mean, r_ple_mean, r_dec_mean, \
                        r_int_sigma,r_ple_sigma,r_dec_sigma)

def rs2score2(rs):
    z = 0
    for kind in ['int','ple','dec']:
        for moment in ['mean','sigma']:
            z += rs[kind][moment]/SIGMAS2[kind+'_'+moment]
    return z/6.0

def z2(kind,moment,predicted,observed): 
    sigma = SIGMAS2[kind+'_'+moment]
    shuffled_r = 0#r2(kind,predicted,shuffled)
    observed_r = r2(kind,moment,predicted,observed)
    return (observed_r - shuffled_r)/sigma

def scorer2(estimator,inputs,observed):
    predicted = estimator.predict(inputs)
    return r2(None,None,predicted,observed)