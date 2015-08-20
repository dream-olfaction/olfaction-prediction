"""Tools for chemical search and ID conversion"""

import urllib

def smile2cid(smile):
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/TXT'
    values = {'smiles' : 'Smiles:%s' % smile}
    data = urllib.parse.urlencode(values)
    data = data.encode('utf-8') # data should be bytes
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as response:
        page = response.read()
    cid = int(page.decode(encoding='UTF-8'))
    return cid

