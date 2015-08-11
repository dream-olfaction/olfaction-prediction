
The files
  nspdk_r3_d4_unaug_gramian.mtx.gz
  nspdk_r3_d4_unaug.svm
are produced by olfaction-prediction/opc_python/degrave/gramian.py, based on a beta version of EDeN.
They contain the unaugmented NSPDK Gramian (kernel matrix) and the corresponding sparse feature vectors respectively.
This software is fully open source and all in Python.

The files
  odors-abag.R3D4.feature.bz2  
  odors-abag.R3D4.matrix.bz2
are produced by the script in Synapse, which requires the (freeware but not open source) software DMax Chemistry Assistant [1]
for functional group augmentation of the molecules [2] and the (open source) C++ implementation of EDeN (which is closer to the original NSPDK implementation [3]).
In QSAR tasks, augmentation sometimes provides a small benefit in predictive power over standard NSPDK.

[1] https://dtai.cs.kuleuven.be/dmax
[1] K. De Grave, F. Costa. Molecular graph augmentation with rings and functional groups. J. Chem. Inf. Mod. 2010.
[2] F. Costa, K. De Grave. Fast neighborhood subgraph pairwise distance kernel. ICML 2010.
