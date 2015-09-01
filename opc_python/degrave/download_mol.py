
import urllib.request
import shutil
import os
import sys

curr_path = os.getcwd()
olfaction_prediction_path = os.path.split(os.path.split(curr_path)[0])[0]
sys.path.append(olfaction_prediction_path)
import opc_python
from opc_python.utils.loading import get_CIDs

mol_path = os.path.join(olfaction_prediction_path, 'data', 'sdf')

def get_challenge_CIDs():
  return sorted(get_CIDs('training') + get_CIDs('leaderboard') + get_CIDs('testset'))

def get_Gabor_CIDs():
  file_path = os.path.join(olfaction_prediction_path,'data', 'derived', 'gabor_CIDs.txt')
  with open(file_path) as f:
    CIDs = sorted(list(set(map(int, f.read().splitlines()))))
  return CIDs

def get_all_CIDs():
  return get_challenge_CIDs() + get_Gabor_CIDs()

#for cid in allCIDs:
#  os.exec('wget -O individual-sdfs/'+cid+'.sdf http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+cid+'/record/SDF/?record_type=2d&response_type=save')

def download_SDF(cid):
  url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+str(cid)+'/record/SDF/?record_type=2d&response_type=save'
  response = urllib.request.urlopen(url)
  data = response.read()
  molStr = data.decode('utf-8')
  return molStr

def download_SDF_to_file(cid):
  url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+str(cid)+'/record/SDF/?record_type=2d&response_type=save'
  file_name = os.path.join(mol_path, 'mol' + str(cid) + '.sdf')
  with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

# This takes a minute or so
def download_all_mol():
  if not os.path.isdir(mol_path):
    os.mkdir(mol_path)
  with open(os.path.join(mol_path, 'all_mol.sdf'), 'w') as accumulator_file:
    for cid in get_all_CIDs():
      print('cid: ' + str(cid))
      file_name = os.path.join(mol_path, 'mol' + str(cid) + '.sdf')
      sdfStr = download_SDF(cid)
      with open(file_name, 'w') as outfile:
        outfile.write(sdfStr)
        accumulator_file.write(sdfStr)

download_all_mol()
