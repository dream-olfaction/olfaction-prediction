import sys
import os

import numpy as np

utils_path = os.path.dirname(os.path.abspath(__file__))
opc_python_path  = os.path.dirname(utils_path)
root_path = os.path.dirname(opc_python_path)
if root_path not in sys.path:
    sys.path.append(root_path)

def prog(num,denom):
    fract = float(num)/denom
    hyphens = int(round(50*fract))
    spaces = int(round(50*(1-fract)))
    sys.stdout.write('\r%.2f%% [%s%s]' % (100*fract,'-'*hyphens,' '*spaces))
    sys.stdout.flush()     


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


class DoubleSS:
    """
    This class produces a new iterator that makes sure that the training and 
    test set do not contains the same molecule at different dilutions, 
    and also that the higher concentration is tested (or 10^-3 for intensity).  
    """
    def __init__(self, ss, col, concs):
        self.splits = ss
        self.col = col
        self.concs = concs

    def __iter__(self):
        for train, test in self.splits:
            train = np.concatenate((2*train,2*train+1))
            if self.col>0:
                test = 2*test+1 # The second (higher) concentration of the pair
            else:
                test = np.concatenate((2*test,2*test+1))
                test = test[self.concs[test]==-3]
            yield train, test
            
    def __len__(self):
        return len(self.splits)