import pdb

import numpy as np
from scipy.stats import pearsonr
from . import loading

from __init__ import *
SIGMAS = {'int': 0.0187,
          'ple': 0.0176,
          'dec': 0.0042}

SIGMAS2 = {'int_mean': 0.1193,
           'ple_mean': 0.1265,
           'dec_mean': 0.0265,
           'int_std': 0.1194,
           'ple_std': 0.1149,
           'dec_std': 0.0281}

# Scoring for sub-challenge 1.
def r(kind,predicted,observed,n_subjects=49,mask=True):
    # Predicted and observed should each be an array of 
    # molecules by perceptual descriptors by subjects
    
    r = 0.0
    for subject in range(1,n_subjects+1):
        p = predicted[subject]
        o = observed[subject]
        try:
            p = p.unstack('Descriptor')
            o = o.unstack('Descriptor')
        except KeyError:
            pass
        r += r2(kind,None,p,o)
    r /= n_subjects
    return r

def score(predicted,observed,n_subjects=49,mask=True,quiet=False):
    r_int = r('int',predicted,observed,mask=mask)
    r_ple = r('ple',predicted,observed,mask=mask)
    r_dec = r('dec',predicted,observed,mask=mask)
    score_ = (z('int',predicted,observed,n_subjects=n_subjects) \
          + z('ple',predicted,observed,n_subjects=n_subjects) \
          + z('dec',predicted,observed,n_subjects=n_subjects))/3
    if not quiet:
        print('Score: %3f; rs = %.3f,%.3f,%.3f' % \
                (score_, r_int, r_ple, r_dec))
    return score_

def rs2score(r_int,r_ple,r_dec):
    z_int = r_int/SIGMAS['int']
    z_ple = r_ple/SIGMAS['ple']
    z_dec = r_dec/SIGMAS['dec']
    return (z_int+z_ple+z_dec)/3.0

def z(kind,predicted,observed,n_subjects=49): 
    std = SIGMAS[kind]
    shuffled_r = 0#r2(kind,predicted,shuffled)
    observed_r = r(kind,predicted,observed,n_subjects=n_subjects)
    return (observed_r - shuffled_r)/std

# Scoring for sub-challenge 2.  

def r2(kind,moment,predicted,observed,quiet=False):
    # Predicted and observed should each be an array of 
    # molecules by 2*perceptual descriptors (means then stds)
    descriptors = loading.get_descriptors(format=True)
    if moment == 'mean':
        try:
            p = predicted['mean']
        except KeyError:
            p = predicted.mean(axis=1)
        try:
            p['Pleasantness']
        except:
            p = p.unstack('Descriptor')
        try:
            o = observed.mean(axis=1).unstack('Descriptor')
        except:
            o = observed.mean(axis=1,level='Descriptor')
    elif moment == 'std':
        try:
            p = predicted['std']
        except KeyError:
            p = predicted.std(axis=1)
        try:
            p['Pleasantness']
        except:
            p = p.unstack('Descriptor')
        try:
            o = observed.std(axis=1).unstack('Descriptor')
        except:
            o = observed.std(axis=1,level='Descriptor')
    elif moment is None:
        p = predicted
        o = observed
        try:
            p = p.unstack('Descriptor')
        except:
            pass
        try:
            o = o.unstack('Descriptor')
        except:
            pass
    else:
        raise ValueError('No such moment: %s' % moment)

    if kind=='int':
        p = p[['Intensity']]
        try:
            o = o[['Intensity']]
        except KeyError:
            if not quiet:
                print("No Intensity information in the observed data.")
            return np.nan
    elif kind=='ple':
        p = p[['Pleasantness']]
        o = o[['Pleasantness']]
    elif kind == 'dec':
        p = p.drop(['Intensity','Pleasantness'],axis=1)
        try:
            o = o.drop(['Intensity','Pleasantness'],axis=1)
        except ValueError:
            o = o.drop(['Pleasantness'],axis=1)
    elif kind in descriptors:
        p = p[[kind]]
        o = o[[kind]]
    elif kind is None:
        p = p
        o = o
    else:
        raise ValueError('No such kind: %s' % kind)
    
    p = p.astype('float64') # In case there are any Nones.  
    
    common_CIDs = list(set(p.index.values).intersection(o.index.values))
    p = p.loc[common_CIDs]
    o = o.loc[common_CIDs]

    r = 0.0
    descriptors = list(p)
    denom = 0.0
    #from IPython.core.debugger import Tracer
    #Tracer()()
    for d in descriptors:
        p_ = p[d]
        o_ = o[d]
        r_ = p_.corr(o_)
        #if kind == 'int':
        #    print(o)
        if ('%f' % r_) != 'nan':
            r += r_
        denom += 1
    if denom == 0.0:
        r = np.nan
    else:
        r /= denom
    return r

def score2(predicted,observed,quiet=False):
    """Final score for sub-challenge 2."""
    score = z2('int','mean',predicted,observed) \
          + z2('ple','mean',predicted,observed) \
          + z2('dec','mean',predicted,observed) \
          + z2('int','std',predicted,observed) \
          + z2('ple','std',predicted,observed) \
          + z2('dec','std',predicted,observed)
    score = score/6.0
    if not quiet:
        r_int_mean = r2('int','mean',predicted,observed,quiet=True)
        r_ple_mean = r2('ple','mean',predicted,observed,quiet=True)
        r_dec_mean = r2('dec','mean',predicted,observed,quiet=True)
        r_int_std = r2('int','std',predicted,observed,quiet=True)
        r_ple_std = r2('ple','std',predicted,observed,quiet=True)
        r_dec_std = r2('dec','std',predicted,observed,quiet=True)
        print(("Subchallenge 2 Score: %.3f\n"
               "\tr_int_mean = %.3f\n"
               "\tr_ple_mean = %.3f\n"
               "\tr_dec_mean = %.3f\n"
               "\tr_int_std = %.3f\n"
               "\tr_ple_std = %.3f\n"
               "\tr_dec_std = %.3f\n"
               % (score, r_int_mean, r_ple_mean, r_dec_mean, 
                  r_int_std,r_ple_std,r_dec_std)))
    return score

def rs2score2(rs):
    z = 0
    for kind in ['int','ple','dec']:
        for moment in ['mean','std']:
            z += rs[kind][moment]/SIGMAS2[kind+'_'+moment]
    return z/6.0

def z2(kind,moment,predicted,observed): 
    std = SIGMAS2[kind+'_'+moment]
    shuffled_r = 0#r2(kind,predicted,shuffled)
    observed_r = r2(kind,moment,predicted,observed)
    return (observed_r - shuffled_r)/std

def scorer2(estimator,inputs,observed):
    predicted = estimator.predict(inputs)
    return r2(None,None,predicted,observed)