

# For easy integration, here we use Fabrizio Costa's NSPDK implementation in Python.
# EDeN is Python 2 only!
# pip2 install git+https://github.com/fabriziocosta/EDeN.git --user

import sys, os
import numpy as np

curr_path = os.getcwd()
olfaction_prediction_path = os.path.split(os.path.split(curr_path)[0])[0]
sys.path.append(olfaction_prediction_path)
import opc_python

def compute_NSPDK_features():
  import eden
  from eden.graph import Vectorizer
  from eden.converter.molecule.obabel import mol_file_to_iterable, obabel_to_eden
  mol_path = olfaction_prediction_path + '/data/sdf/'
  iter_mols = mol_file_to_iterable(mol_path + '/all_mol.sdf', 'sdf')
  iter_graphs = obabel_to_eden(iter_mols)

  vectorizer = Vectorizer( r=3, d=4 )
  X = vectorizer.transform( iter_graphs )
  return X

def gramian(features):
  from sklearn import metrics
  K=metrics.pairwise.pairwise_kernels(X, metric='linear')
  return K

def plot_square_matrix_heatmap(matrix):
  # %matplotlib inline
  import pylab as plt
  plt.figure( figsize=(15,15) )
  img = plt.imshow( matrix, interpolation='none', cmap=plt.get_cmap( 'YlOrRd' ) )
  plt.show()

def write_to_file(X,K):
  from scipy import io
  from sklearn.datasets.svmlight_format import dump_svmlight_file
  dd_path = olfaction_prediction_path + '/data/derived/'
  if not os.path.isdir(dd_path):
    os.mkdir(dd_path)
  np.savetxt(dd_path + 'nspdk_r3_d4_unaug_gramian.mtx.gz', K)
  # Write features in standard libSVM format:
  dump_svmlight_file(X,np.zeros(X.shape[0]),dd_path + 'nspdk_r3_d4_unaug.svm')
  # Alternatively, write one element per line:
  #io.mmwrite(dd_path + 'nspdk.features', X)

X = compute_NSPDK_features()
K = gramian(X)
plot_square_matrix_heatmap(K)
write_to_file(X,K)
