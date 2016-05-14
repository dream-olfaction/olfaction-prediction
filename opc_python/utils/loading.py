import os, sys, csv, ast, time, json

import numpy as np

from __init__ import *
import opc_python
from opc_python import * # Import constants.  
ROOT_PATH = os.path.split(opc_python.__path__[0])[0]
from opc_python.utils import scoring,search

DATA_PATH = os.path.join(ROOT_PATH,'data')
PREDICTION_PATH = os.path.join(ROOT_PATH,'predictions')

def load_perceptual_data(kind, just_headers=False):
    if kind in ['training','training-norep','replicated']:
        kind2 = 'TrainSet'
    elif kind == 'leaderboard':
        kind2 = 'LeaderboardSet'
    elif kind == 'testset':
        kind2 = 'TestSet'
    else:
        raise ValueError("No such kind: %s" % kind)
    
    if kind in ['training-norep','replicated']:
        with_replicates = get_CID_dilutions('replicated')
    data = []
    file_path = os.path.join(DATA_PATH,'%s.txt' % kind2)
    with open(file_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                line[0:6] = [x.strip() for x in line[0:6]]
                line[2] = True if line[2]=='replicate' else False
                line[6:] = ['NaN' if x=='NaN' else int(x) for x in line[6:]]
                CID = int(line[0])
                dilution = line[4]
                mag = dilution2magnitude(dilution)
                high = line[3] == 'high'
                CID_dilution = "%d_%g_%d" % (CID,mag,high)
                if kind == 'training-norep':
                    if CID_dilution not in with_replicates:
                        data.append(line)
                elif kind == 'replicated':
                    if CID_dilution in with_replicates:
                        data.append(line)
                else:
                    data.append(line)
            else:
                headers = line
                if just_headers:
                    return headers
    return headers,data


def get_descriptors():
    headers = load_perceptual_data('training', just_headers=True)
    return headers[6:]


def format_leaderboard_perceptual_data(data_path=''):
    new_leaderboard_file_path = os.path.join(DATA_PATH,'LeaderboardSet.txt')
    f_new = open(new_leaderboard_file_path,'w')
    writer = csv.writer(f_new,delimiter="\t")
    headers,_ = load_perceptual_data('training')
    descriptors = headers[6:]
    writer.writerow(headers)
    CID_dilutions = get_CID_dilutions('leaderboard',target_dilution='raw')
    dilutions = {}
    for CID_dilution in CID_dilutions:
        CID,mag,high = CID_dilution.split('_')
        dilutions[int(CID)] = {'dilution':('1/%d' % 10**(-int(mag))),
                               'high':int(high)}
    lines_new = {}
    
    lbs1_path = os.path.join(DATA_PATH,'LBs1.txt')
    with open(lbs1_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                CID,subject,descriptor,value = line
                CID = int(CID)
                subject = int(subject)
                if descriptor == 'INTENSITY/STRENGTH':
                    dilution = '1/1000'
                    high = 1-dilutions[CID]['high']
                else:
                    high = dilutions[CID]['high']
                    dilution = dilutions[CID]['dilution'] if high else '1/1000'
                line_id = '%d_%d_%d' % (CID,subject,dilution2magnitude(dilution))
                if line_id not in lines_new:
                    lines_new[line_id] = [CID,'N/A',0,'high' if high else 'low',dilution,subject]+['NaN']*21
                lines_new[line_id][6+descriptors.index(descriptor.strip())] = value

    leaderboard_set_file_path = os.path.join(DATA_PATH,'leaderboard_set.txt')
    with open(leaderboard_set_file_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                CID,subject,descriptor,value = line
                CID = int(CID)
                subject = int(subject)
                if descriptor == 'INTENSITY/STRENGTH':
                    dilution = dilutions[CID]['dilution']
                    high = dilutions[CID]['high']
                else:
                    high = 1-dilutions[CID]['high']
                    dilution = dilutions[CID]['dilution'] if high else '1/1000'
                line_id = '%d_%d_%d' % (CID,subject,dilution2magnitude(dilution))
                if line_id not in lines_new:
                    lines_new[line_id] = [CID,'N/A',0,'high' if high else 'low',dilution,subject]+['NaN']*21
                lines_new[line_id][6+descriptors.index(descriptor.strip())] = value


    for line_id in sorted(lines_new,key=lambda x:[int(_) for _ in x.split('_')]):
        line = lines_new[line_id]
        writer.writerow(line)
    f_new.close()

def format_testset_perceptual_data(data_path=''):
    new_leaderboard_file_path = os.path.join(DATA_PATH,'TestSet.txt')
    f_new = open(new_leaderboard_file_path,'w')
    writer = csv.writer(f_new,delimiter="\t")
    headers,_ = load_perceptual_data('training')
    descriptors = headers[6:]
    writer.writerow(headers)
    CID_dilutions = get_CID_dilutions('testset',target_dilution='raw')
    dilutions = {}
    for CID_dilution in CID_dilutions:
        CID,mag,high = CID_dilution.split('_')
        dilutions[int(CID)] = {'dilution':('1/%d' % 10**(-int(mag))),
                               'high':int(high)}
    lines_new = {}
    
    gs_path = os.path.join(DATA_PATH,'GS.txt')
    with open(gs_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                CID,subject,descriptor,value = line
                CID = int(CID)
                subject = int(subject)
                if descriptor == 'INTENSITY/STRENGTH':
                    dilution = '1/1000'
                    high = 1-dilutions[CID]['high']
                else:
                    high = dilutions[CID]['high']
                    dilution = dilutions[CID]['dilution'] if high else '1/1000'
                line_id = '%d_%d_%d' % (CID,subject,dilution2magnitude(dilution))
                if line_id not in lines_new:
                    lines_new[line_id] = [CID,'N/A',0,'high' if high else 'low',dilution,subject]+['NaN']*21
                lines_new[line_id][6+descriptors.index(descriptor.strip())] = value

    for line_id in sorted(lines_new,key=lambda x:[int(_) for _ in x.split('_')]):
        line = lines_new[line_id]
        writer.writerow(line)
    f_new.close()


def load_leaderboard_perceptual_data(target_dilution=None):
    """Loads directly into Y"""
    if target_dilution == 'gold':
        # For actual testing, use 1/1000 dilution for intensity and
        # high dilution for everything else.  
        Y = load_leaderboard_perceptual_data(target_dilution='high')
        intensity = load_leaderboard_perceptual_data(target_dilution=-3)
        Y['mean_std'][:,0] = intensity['mean_std'][:,0]
        Y['mean_std'][:,21] = intensity['mean_std'][:,21]
        for i in range(len(Y['subject'])):
            Y['subject'][i][:,0] = intensity['subject'][i][:,0]
            Y['subject'][i][:,21] = intensity['subject'][i][:,21]
        return Y
    #assert target_dilution in [None,'low','high']
    perceptual_headers, _ = load_perceptual_data()
    descriptors = perceptual_headers[6:]
    CIDs = get_CID_dilutions('leaderboard',target_dilution=target_dilution)
    CIDs_all = get_CID_dilutions('leaderboard',target_dilution=None)
    CIDs = [int(_.split('_')[0]) for _ in CIDs] # Keep only CIDs, not dilutions.  
    n_molecules = len(CIDs)
    Y = {#'mean':np.zeros((n_molecules,21)),
         #'std':np.zeros((n_molecules,21)),
         'mean_std':np.zeros((n_molecules,42)),
         'subject':{i:np.zeros((n_molecules,21)) for i in range(1,50)}}
    if target_dilution not in ['low','high',None]:
        CID_ranks = get_CID_rank('leaderboard',dilution=target_dilution)
    if target_dilution != 'low':
        lbs1_path = os.path.join(DATA_PATH,'LBs1.txt')
        with open(lbs1_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num,line in enumerate(reader):
                if line_num > 0:
                    CID,subject,descriptor,value = line
                    if target_dilution not in ['low','high',None] and CID_ranks[int(CID)]!=1:
                        continue
                    Y['subject'][int(subject)][CIDs.index(int(CID)),descriptors.index(descriptor.strip())] = value
        
        lbs2_path = os.path.join(DATA_PATH,'LBs2.txt')
        with open(lbs2_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num,line in enumerate(reader):
                if line_num > 0:
                    CID,descriptor,mean,std = line
                    if target_dilution not in ['low','high',None] and CID_ranks[int(CID)]!=1:
                        continue
                    #Y['mean'][CIDs.index(CID),descriptors.index(descriptor.strip())] = mean
                    #Y['std'][CIDs.index(CID),descriptors.index(descriptor.strip())] = std
                    Y['mean_std'][CIDs.index(int(CID)),descriptors.index(descriptor.strip())] = mean
                    Y['mean_std'][CIDs.index(int(CID)),21+descriptors.index(descriptor.strip())] = std
                    
    if target_dilution != 'high':
        leaderboard_set_file_path = os.path.join(DATA_PATH,'leaderboard_set.txt')
        with open(leaderboard_set_file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num,line in enumerate(reader):
                if line_num > 0:
                    CID,subject,descriptor,value = line
                    if target_dilution not in ['low','high',None] and CID_ranks[int(CID)]!=0:
                        continue
                    Y['subject'][int(subject)][CIDs.index(int(CID)),descriptors.index(descriptor.strip())] = value
            z = np.dstack([_ for _ in Y['subject'].values()])
            mask = np.zeros(z.shape)
            mask[np.where(np.isnan(z))] = 1
            z = np.ma.array(z,mask=mask)
            y_mean = z.mean(axis=2)
            y_std = z.std(axis=2,ddof=1)
        for CID in CID_ranks:
            if CID_ranks[CID] == 0:
                row = CIDs.index(CID)
                col = descriptors.index(descriptor.strip())
                Y['mean_std'][row,:21] = y_mean[row,:]
                Y['mean_std'][row,21:] = y_std[row,:]
    Y['mean_std'] = Y['mean_std'].round(2)
    return Y

def load_molecular_data():
    mdd_file_path = os.path.join(DATA_PATH,'molecular_descriptors_data.txt')
    with open(mdd_file_path) as f:
        reader = csv.reader(f, delimiter="\t")
        data = []
        for line_num,line in enumerate(reader):
            if line_num > 0:
                line[1:] = ['NaN' if x=='NaN' else float(x) for x in line[1:]]
                data.append(line)
            else:
                headers = line
    return headers,data

def get_CID_dilutions(kind,target_dilution=None):
    assert kind in ['training','training-norep','replicated','leaderboard','testset']
    """Return CIDs for molecules that will be used for:
        'leaderboard': the leaderboard to determine the provisional 
                       leaders of the competition.
        'testset': final testing to determine the winners 
                   of the competition."""
    if kind in ['training','replicated']:
        data = []
        _,lines = load_perceptual_data('training')
        for line in lines[1:]:
            if line[2] or kind != 'replicated':
                CID = int(line[0])
                dilution = line[4]
                mag = dilution2magnitude(dilution)
                high = line[3] == 'high'
                if target_dilution == 'high' and not high:
                    continue
                if target_dilution == 'low' and not low:
                    continue
                elif target_dilution not in [None,'high','low'] and \
                     mag != target_dilution:
                     continue
                data.append("%d_%g_%d" % (CID,mag,high))
        data = list(set(data))
    elif kind == 'training-norep':
        training = set(get_CID_dilutions('training',target_dilution=target_dilution))
        replicated = set(get_CID_dilutions('replicated',target_dilution=target_dilution))
        data = list(training.difference(replicated))
    else:
        dilution_file_path = os.path.join(DATA_PATH,'dilution_%s.txt' % kind)
        with open(dilution_file_path) as f:
            reader = csv.reader(f,delimiter='\t')
            next(reader)
            lines = [[int(line[0]),line[1]] for line in reader]
            data = []
            for i,(CID,dilution) in enumerate(lines):
                mag = dilution2magnitude(dilution)
                high = (mag > -3)
                if target_dilution in ['high','gold']:
                    if high:
                        data.append('%d_%g_%d' % (CID,mag,1))
                    else:
                        data.append('%d_%g_%d' % (CID,-3,1))
                elif target_dilution == 'low':
                    if not high:
                        data.append('%d_%g_%d' % (CID,mag,0))
                    else:
                        data.append('%d_%g_%d' % (CID,-3,0))
                elif target_dilution is None:
                    data.append('%d_%g_%d' % (CID,mag,high))
                    data.append('%d_%g_%d' % (CID,-3,1-high))
                elif target_dilution in [mag,'raw']:
                    data.append('%d_%g_%d' % (CID,mag,high))
                elif target_dilution == -3:
                    data.append('%d_%g_%d' % (CID,-3,1-high))
 
    return sorted(data,key=lambda x:[int(_) for _ in x.split('_')])

def get_CIDs(kind):
    CID_dilutions = get_CID_dilutions(kind)
    CIDs = [int(_.split('_')[0]) for _ in CID_dilutions]
    return sorted(list(set(CIDs)))

def get_CID_rank(kind,dilution=-3):
    """Returns CID dictionary with 1 if -3 dilution is highest, 
    0 if it is lowest, -1 if it is not present.
    """

    CID_dilutions = get_CID_dilutions(kind)
    CIDs = set([int(_.split('_')[0]) for _ in CID_dilutions])
    result = {}
    for CID in CIDs:
        high = '%d_%g_%d' % (CID,dilution,1)
        low = '%d_%g_%d' % (CID,dilution,0)
        if high in CID_dilutions:
            result[CID] = 1
        elif low in CID_dilutions:
            result[CID] = 0
        else:
            result[CID] = -1
    return result

def dilution2magnitude(dilution):
    denom = dilution.replace('"','').replace("'","").split('/')[1].replace(',','')
    return np.log10(1.0/float(denom))


def load_data_matrix(kind='training',gold_standard_only=False,
                     only_replicates=False):
    """
    Loads the data into a 6-dimensional matrix:
    Indices are:
     subject number (0-48)
     CID index
     descriptor number (0-20)
     dilution rank (1/10=0, 1/1000=1, 1/100000=2, 1/1000000=3)
     replicate (original=0, replicate=1)
    Data is masked so locations with no data are not included in statistics on this array.  
    """

    _, perceptual_obs_data = load_perceptual_data(kind)
    CIDs = get_CIDs(kind)
    data = np.ma.zeros((49,len(CIDs),21,4,2),dtype='float')
    data.mask +=1
    for line in perceptual_obs_data:
        CID_index = CIDs.index(int(line[0]))
        subject = int(line[5])
        is_replicate = line[2]
        #is_replicate = False
        dilution_index = ['1/10','1/1,000','1/100,000','1/10,000,000'].index(line[4])
        for i,value in enumerate(line[6:]):
            indices = subject-1,CID_index,i,dilution_index,int(is_replicate)
            if value != 'NaN':
                if gold_standard_only:
                    if (i==0 and dilution_index==1) or (i>0 and line[3]=='high'):
                        data[indices] = float(value)
                else:
                    data[indices] = float()
    if only_replicates:
        only_replicates = data.copy()
        only_replicates.mask[:,:,:,:,0] = (data.mask[:,:,:,:,0] + data.mask[:,:,:,:,1])>0
        only_replicates.mask[:,:,:,:,1] = (data.mask[:,:,:,:,0] + data.mask[:,:,:,:,1])>0
        data = only_replicates
    return data


"""Output"""

# Write predictions for each subchallenge to a file.  
def open_prediction_file(subchallenge,kind,name):
    prediction_file_path = os.path.join(PREDICTION_PATH,\
                                        'challenge_%d_%s_%s.txt' \
                                        % (subchallenge,kind,name))
    f = open(prediction_file_path,'w')
    writer = csv.writer(f,delimiter='\t')
    return f,writer

def write_prediction_files(Y,kind,subchallenge,name):
    f,writer = open_prediction_file(subchallenge,kind,name=name)
    CIDs = get_CIDs(kind)
    perceptual_headers, _ = load_perceptual_data('training')

    # Subchallenge 1.
    if subchallenge == 1:
        writer.writerow(["#oID","individual","descriptor","value"])
        for subject in range(1,NUM_SUBJECTS+1):
            for j in range(NUM_DESCRIPTORS):
                for i,CID in enumerate(CIDs):
                    descriptor = perceptual_headers[-NUM_DESCRIPTORS:][j]
                    value = Y['subject'][subject][i,j].round(3)
                    writer.writerow([CID,subject,descriptor,value])
        f.close()
    
    # Subchallenge 2.
    elif subchallenge == 2:
        writer.writerow(["#oID","descriptor","value","sigma"])
        for j in range(NUM_DESCRIPTORS):
            for i,CID in enumerate(CIDs):
                descriptor = perceptual_headers[-NUM_DESCRIPTORS:][j]
                value = Y['mean_std'][i,j].round(3)
                sigma = Y['mean_std'][i,j+NUM_DESCRIPTORS].round(3)
                writer.writerow([CID,descriptor,value,sigma])
        f.close()

def make_prediction_files(rfcs,X_int,X_other,target,subchallenge,Y_test=None,
                          write=True,trans_weight=np.zeros(21),
                          trans_params=np.ones((21,2)),regularize=[0.8],name=None):
    if len(regularize)==1 and type(regularize)==list:
        regularize = regularize*21
    if name is None:
        name = '%d' % time.time()

    Y = {'subject':{}}
    
    if subchallenge == 1:
        y0 = rfcs[0][1].predict(X_int)
        ys = np.zeros((y0.shape[0],21,49))
        for col in range(21):
            X = X_int if col==0 else X_other
            for subject in range(1,50):
                ys[:,col,subject-1] = rfcs[col][subject].predict(X)
        
        # Regularize
        ys_mean = ys.mean(axis=2,keepdims=True)
        ys_reg = ys.copy()
        for col in range(21):
            ys_reg[:,col,:] = regularize[col]*ys_mean[:,col,:] + (1-regularize[col])*ys[:,col,:]
        ys = ys_reg

        for subject in range(1,50):
            Y['subject'][subject] = ys[:,:,subject-1]
        
        if Y_test:
            predicted = ys.copy()
            observed = ys.copy()
            for subject in range(1,50):
                predicted[:,:,subject-1] = Y['subject'][subject]
                observed[:,:,subject-1] = Y_test['subject'][subject]
            print(scoring.score_summary(predicted,observed))
            
    if subchallenge == 2:
        def f_transform(x, k0, k1):
            return 100*(k0*(x/100)**(k1*0.5) - k0*(x/100)**(k1*2))
        
        kinds = ['int','ple','dec']
        moments = ['mean','sigma']
        y = np.zeros((X_int.shape[0],42))
        for col in range(42):
            X = X_int if col in (0,21) else X_other
            y[:,col] = rfcs[col].predict(X)
        
        for col in range(21):
            tw = trans_weight[col]
            k0,k1 = trans_params[col]
            y_m = y[:,col]
            y_s = y[:,col+21]
            y[:,col+21] = tw*f_transform(y_m,k0,k1) + (1-tw)*y_s
        
        Y['mean_std'] = y
        if Y_test is not None:
            print(scoring.score_summary2(Y['mean_std'],Y_test['mean_std'],mask=True))
            
    if write:
        write_prediction_files(Y,target,subchallenge,name)
        print('Wrote to file with suffix "%s"' % name)
    return Y

def load_eva_data(save_formatted=False):
    eva_file_path = os.path.join(DATA_PATH,'eva_100_training_data.json')
    with open(eva_file_path) as f:
        eva_json = json.load(f)

    smile_cids = {}
    for smile in eva_json.keys():
        smile_cids[smile] = search.smile2cid(smile)

    cid_smiles = {cid:smile for smile,cid in smile_cids.items()}
    eva_cids = list(smile_cids.values())
    available_cids = []
    eva_data = []
    for kind in ('training','leaderboard','testset'):
        dream_cids = get_CIDs(kind)
        print('Out of %d CIDs from the %s data, we have EVA data for %d of them.' \
            % (len(dream_cids),kind,len(set(dream_cids).intersection(eva_cids))))
    for cid in dream_cids:
        if cid in cid_smiles:
            available_cids.append(cid)

    available_cids = sorted(available_cids)
    for cid in available_cids:
        smile = cid_smiles[cid]
        eva_data.append(eva_json[smile])

    if save_formatted:
        np.savetxt(os.path.join(DATA_PATH,'derived','eva_cids.dat'),
                   available_cids)
        np.savetxt(os.path.join(DATA_PATH,'derived','eva_descriptors.dat'),
                   eva_data)

    return (available_cids,eva_data)