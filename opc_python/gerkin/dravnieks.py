"""
dravieks.txt produced by copying and pasting from the supplemental material of 
"In search of the structure of human olfactory space", e.g. koulakov.pdf.  
"""

import os

DATA_PATH = '../../data'
n_subjects = 150 # Number of subjects reflected in this data set. Unconfirmed.

with open(os.path.join(DATA_PATH,'dravnieks_cas.txt'),'r') as f:
    text = f.read()
    lines = text.split('\n')[1:]
    cas = [line.split(' ')[1] for line in lines] # A list of CAS numbers.  

with open(os.path.join(DATA_PATH,'dravnieks_descriptors.txt'),'r') as f:
    text = f.read()
    lines = text.split('\n')[1:]
    descriptors = [' '.join(line.split(' ')[1:]) for line in lines] # A list of descriptors.

with open(os.path.join(DATA_PATH,'dravnieks_data.txt'),'r') as f:
    text = f.read()
    lines = text.split('\n')[:-1]
    data = {}
    for i,line in enumerate(lines):
        values = line.split('\t')
        data[cas[i]] = {descriptors[j]:float(value)/n_subjects for j,value in enumerate(values)}




    