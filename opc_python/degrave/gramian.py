

# For easy integration, here we use Fabrizio Costa's NSPDK implementation in Python.
# EDeN is Python 2 only!
# pip2 install git+https://github.com/fabriziocosta/EDeN.git --user

import sys, os
import numpy as np

import eden
from eden.graph import Vectorizer
from eden.converter.molecule.obabel import mol_file_to_iterable, obabel_to_eden

curr_path = os.getcwd()
olfaction_prediction_path = os.path.split(os.path.split(curr_path)[0])[0]
sys.path.append(olfaction_prediction_path)
import opc_python

mol_path = olfaction_prediction_path + '/data/sdf/'

iter_mols = mol_file_to_iterable(mol_path + '/all_mol.sdf', 'sdf')
iter_graphs = obabel_to_eden(iter_mols)

vectorizer = Vectorizer( r=3, d=4 )
X = vectorizer.transform( iter_graphs )

# %matplotlib inline
from sklearn import metrics
K=metrics.pairwise.pairwise_kernels(X, metric='linear')
print K

import pylab as plt
plt.figure( figsize=(8,8) )
img = plt.imshow( K, interpolation='none', cmap=plt.get_cmap( 'YlOrRd' ) )
plt.show()
