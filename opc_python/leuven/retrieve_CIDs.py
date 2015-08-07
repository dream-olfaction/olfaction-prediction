# Leander Schietgat (KU Leuven)
# 7 August 2015

import re

# 1. Retrieve training set CIDs
# Caution: TrainSet.txt contains ^r at end of every line rather than expected \n, so Python sees only one line

train_file = open ('../../data/TrainSet.txt','r')
new_train_file = open ('../../data/TrainSet_corrected.txt','w')

line = train_file.readline()
line = line.replace('\r','\n')
new_train_file.write(line)

train_file.close()
new_train_file.close()

train_file = open ('../../data/TrainSet_corrected.txt','r')
train_cid = open ('CID_train','w')

train_ids = set([])

for line in train_file:
  if not line.startswith('Compound Identifier'):
    cells = line.split('\t')
    train_ids.add(cells[0])

for id in sorted(train_ids, key=int):
  train_cid.write(id+'\n')

train_file.close()
train_cid.close()

# 2. Retrieve leaderboard CIDs

leaderboard_file = open ('../../data/CID_leaderboard.txt','r')
leaderboard_cid = open ('CID_leaderboard','w')

for line in leaderboard_file:
  m = re.search('^\s?(\d+)',line)
  leaderboard_cid.write(m.group(1)+'\n')

leaderboard_file.close()
leaderboard_cid.close()

# 3. Retrieve test set CIDs

testset_file = open ('../../data/CID_testset.txt','r')
testset_cid = open ('CID_testset','w')

for line in testset_file:
  m = re.search('^\s?(\d+)',line)
  testset_cid.write(m.group(1)+'\n')

testset_file.close()
testset_cid.close()

