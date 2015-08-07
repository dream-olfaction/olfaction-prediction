
import urllib.request
import shutil
import os
import sys
curr_path = os.getcwd()
olfaction_prediction_path = os.path.split(os.path.split(curr_path)[0])[0]
sys.path.append(olfaction_prediction_path)
import opc_python

from opc_python import * # Import constants.  
from opc_python.utils.loading import get_CIDs

allCIDs = sorted(get_CIDs('training') + get_CIDs('leaderboard') + get_CIDs('testset'))

mol_path = olfaction_prediction_path + '/data/sdf/'
if not os.path.isdir(mol_path):
  os.mkdir(mol_path)

#for cid in allCIDs:
  #os.exec('wget -O individual-sdfs/${CID}.sdf http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${CID}/record/SDF/?record_type=2d&response_type=save')

def download_SDF(cid):
  url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+str(cid)+'/record/SDF/?record_type=2d&response_type=save'
  response = urllib.request.urlopen(url)
  data = response.read()
  molStr = data.decode('utf-8')
  return molStr

def download_SDF_to_file(cid):
  url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+str(cid)+'/record/SDF/?record_type=2d&response_type=save'
  file_name = mol_path + 'mol' + str(cid) + '.sdf'
  with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

for cid in allCIDs:
  download_SDF_to_file(cid)
