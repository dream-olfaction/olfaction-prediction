function Eval_LB()
clear;

load 'train_set.mat';


%%
% reduce molecular descriptors' dimensionality

molecular = MolecularNum;

[i j] = find( isnan(molecular) );

molecular(i,j)=0;


%delta = 0.001;
%minval = min(min(molecular(:,2:end))')

molecular(:,2:end) = log(100+molecular(:,2:end));

%%

% %%%% TRAINING %%%% %

% choose appropriate trials:
% (1) eliminate 'non responses'
% (2) keep only 'high' concentrations
% (3) identify 1/1,000 concentrations to test Intensity
% (4) identify odors with replicates and without 1/1,000 trials
%     to keep strictly in the training set

% subjects
sjID = unique(MolecularNum(:,6));

% good trials
I1 = find( sum(isnan( trainNum(:,[1 6:27]) )') == 0 )';   

I2 = find( strcmp( trainTxt( 2:end,4), 'high' ) );

goodtrials = intersect( I1, I2);

features = trainNum( goodtrials, [1 6:27]);

% trials for testing Intensity
I3 = find(strcmp(trainTxt(2:end,5),'1/1,000')); 

trialsForIntens = intersect( I1, I3);

featIntens =  trainNum( trialsForIntens, [1 6:7]);

% find odors without 1/1,000 instances: 
I = find(strcmp(trainTxt(:,5),'1/1,000')); 
withOneThou = unique(trainNum(I-1,1));
noOneThou = setdiff( odorsID(trainIdx), withOneThou);

noOT_list = [];
for i=1:length(noOneThou)
    n = find(odorsID(trainIdx)==noOneThou(i));
    noOT_list = [noOT_list n];
end

% build perceptual matrix, averaging over subjects:

K = zeros(length(trainIdx),21); % or 19 or 21

for i=1:length(trainIdx), 
    I=find(features(:,1)==odorsID(trainIdx(i)) ); 
    % mean of features for each odor
    K(i,:) = std(features(I,3:end),1); % 3:end or 5:end
%     K(i,:) = mean(features(I,3:end),1); % 3:end or 5:end
    
    % intensity:
    I=find(featIntens(:,1)==odorsID(trainIdx(i)) );    
    K(i,1) = std(featIntens(I,3),1);
%     K(i,1) = mean(featIntens(I,3),1);
end;

% intensity for odors without 1/1,000
for i=1:length(noOneThou)
    % I1: all good trials
    I4 = find( trainNum(:,1) == noOneThou(i) );
    I5 = intersect( I1, I4 );
%     K( noOT_list(i), 1 ) = mean( trainNum(I5,7) );
    K( noOT_list(i), 1 ) = std( trainNum(I5,7) );
end

%LBs2 = dlmread('LBs2.txt','\t',1,2);

%K = [K;zeros(69,21)];
%for i=1:21
 %   startindex = (i-1)*69+1;
  %  lastindex = i*69;

   % K(339:end,i) = LBs2(startindex:lastindex,2);
%end

proj_std = dlmread('Subchallenge2FTstepwise100linADJPredsStd.csv',',',0,1);

proj_std_tr = dlmread('Subchallenge2Trainstepwise100linADJPredsStd.csv',',',0,1);


%base_std = dlmread('Subchallenge2FTBasePredictionsEntire_std.csv',',');
base_std = dlmread('CecchiG_DREAM95olf_s2_cleaned.txt','\t',1,3);

base_std_tr = dlmread('Subchallenge2TrainBasePredictions_std.csv',',');
%base_std_tr = dlmread('Subchallenge2TrainBasePredictionsEntire_std.csv',',');

proj_std = max(0,min(100,proj_std));

proj_std_tr = max(0,min(100,proj_std_tr));

%proj_std(:,2) = min(proj_std(:,2),base_std(:,2));
proj_std(:,2) = min(proj_std(:,2),base_std(70:138));
proj_std_tr(:,2) = max(proj_std_tr(:,2),base_std_tr(:,2));

scaledVstd = zeros(69,1);

minVtr = min(K(:,2));
maxVtr = max(K(:,2));
meanVtr = mean(K(:,2));
minVpred = min(proj_std_tr(:,2));
maxVpred = max(proj_std_tr(:,2));

for i=1:69
    scaledVstd(i) = proj_std(i,2);
    if minVpred*minVtr > 0
        if proj_std(i,2) < meanVtr
            scaledVstd(i) = proj_std(i,2)*minVtr/minVpred;
        end    
    elseif proj_std(i,2) > meanVtr
        scaledVstd(i) = proj_std(i,2)*maxVtr/maxVpred;
    end
end

dlmwrite('Subchallenge2FTPredsStdV_Final1.csv',[odorsID(testIdx) scaledVstd],'precision',10);

