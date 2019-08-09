import sys
import os
from itertools import starmap

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

utils_path = os.path.dirname(os.path.abspath(__file__))
opc_python_path = os.path.dirname(utils_path)
root_path = os.path.dirname(opc_python_path)
if root_path not in sys.path:
    sys.path.append(root_path)

def prog(num,denom,msg=''):
    if msg:
        msg += ' '
    fract = float(num)/denom
    hyphens = int(round(50*fract))
    spaces = int(round(50*(1-fract)))
    sys.stdout.write('\r%s%.2f%% [%s%s]' % (msg,100*fract,'-'*hyphens,' '*spaces))
    sys.stdout.flush()     


class ProgressBar:
    """
    An animated progress bar used like:
    n = 1000
    p = ProgressBar(n)
    for i in range(n):
        p.animate(i)
        # Or using optional message:
        # p.animate(i,"{k} out of {n} complete")
        time.sleep(1) # Replaced by your code
    p.animate(n)
    """
    
    def __init__(self, n, progress='{k} out of {n} complete', status=''):
        self.n = n
        self.progress = progress
        self.status = status
        self.char = '-'
        self.width = 50
        self.last_length = 0
        self._update(0, '')

    def animate(self, k, status=None):
        k_ = self.n if k is None else k
        if status is None:
            status = self.status
        sys.stdout.write('\r%s' % (' '*self.last_length))
        self._update(k_, status)
        sys.stdout.write('\r%s' % self.text)
        if k is None:
            sys.stdout.write('\n')
        sys.stdout.flush()
        
    def _update(self, k, status):
        percent = int(k*100/self.n)
        n_chars = int(round((percent / 100.0) * (self.width-2)))
        self.text = "[%s%s]" % (self.char*n_chars,' '*(self.width-2-n_chars))
        pct = ('%d%%' % percent)
        self.text = self.text[:int(self.width/2)-1] + pct + self.text[int(self.width/2)-1+len(pct):]
        self.text += ' '+self.progress.format(k=k,n=self.n)
        if status:
            self.text += ' (%s)' % status
        self.last_length = len(self.text)


# Single-letter codes for each descriptor.  
codes = ['Intensity',
'Valence/pleasantness',
'Bakery',
'swEet',
'fRuit',
'Fish',
'garLic',
'sPices',
'Cold',
'Sour',
'burNt',
'Acid',
'Warm',
'musKy',
'sweaTy',
'ammonia/Urinous',
'decaYed',
'wooD',
'Grass',
'flOwer',
'cheMical']
letters = [[x for x in code if (x.upper() == x and x!='/') ][0] for code in codes]

class DreamGroupShuffleSplit(GroupShuffleSplit):
    def split(self, X, CIDs, desc):
        # CIDs should be a list or index of CIDs with a length equal to the
        # number of entries in the index of X.

        def f(train, test):
            train = X.index[list(train)]
            test = X.index[list(test)]
            X_test = X.loc[test]
            if desc == 'Intensity':
                # If concentration is 10^-3
                test = [z for z in test if z[1] == -3]
            else:
                test = X_test.groupby(level=['CID', 'Dilution']).last().index
            return train, test

        gss = super(DreamGroupShuffleSplit, self).split(X, groups=CIDs)
        return starmap(f, gss)


class DoubleSS:
    """
    This class produces a new iterator that makes sure that the training and
    test set do not contains the same molecule at different dilutions,
    and also that the higher concentration is tested (or 10^-3 for intensity).

    Deprecated in favor of DreamGroupShuffleSplit which works in one step.
    """
    def __init__(self, ss, n_obs, col, concs):
        self.splits = ss
        self.n_obs = n_obs
        self.col = col
        self.concs = concs

    def __iter__(self):
        for train, test in self.splits.split(range(self.n_obs)):
            train = np.concatenate((2*train, 2*train+1))
            if self.col > 0:
                # The second (higher) concentration of the pair
                test = 2*test + 1
            else:
                test = np.concatenate((2*test, 2*test+1))
                test = test[self.concs[test] == -3]
            yield train, test

    def __len__(self):
        return len(self.splits)
