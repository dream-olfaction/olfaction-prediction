%%
%
% read train and test/leader, learn, predict and evaluate
%

clear

load 'train_set.mat';
% includes:
% testIdx 
% leadIdx 
% trainIdx 
% odorsID 
% trainNum 
% trainTxt 
% MolecularNum 
% MolecularTxt


% % % % % % % % % % % % % % % % % % % %
% perceptual features in *Num matrices:
% '1'    'component identifier'
% '2'    'Odor'
% '3'    'replicate'
% '4'    'intensity'
% '5'    'dilution'
% '6'    'subject #'
% '7'    'INTENSITY/STRENGTH'
% '8'    'VALENCE/PLEASANTNESS '
% '9'    'BAKERY'
% '10'    'SWEET'
% '11'    'FRUIT'
% '12'    'FISH'
% '13'    'GARLIC'
% '14'    'SPICES'
% '15'    'COLD'
% '16'    'SOUR'
% '17'    'BURNT'
% '18'    'ACID'
% '19'    'WARM'
% '20'    'MUSKY'
% '21'    'SWEATY'
% '22'    'AMMONIA/URINOUS'
% '23'    'DECAYED'
% '24'    'WOOD'
% '25'    'GRASS'
% '26'    'FLOWER'
% '27'    'CHEMICAL'
% % % % % % % % % % % % % % % % % % % %

%%
% settings

% set to 1 to remove the mean of perceptual values
wM = 0;

% poly order; best is 3
ORD = 3;

% set to 1 for pearson, 0 for spearman
usePearson = 1;

% dimensionality of reduced molecular descriptors
% 30 to 40 is a good range
pcN = 40;

if usePearson
    myCorrType = 'Pearson';
else
    myCorrType = 'Spearman';
end

%%
% reduce molecular descriptors' dimensionality

molecular = MolecularNum;

[i j] = find( isnan(molecular) );

molecular(i,j)=0;

%[U S V] = svd(log(100+molecular),'econ');

molecular(:,2:end) = log(100+molecular(:,2:end));

%dlmwrite('LogInput.csv',molecular,'precision',10);

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


% build perceptual matrix:

K = zeros(length(trainIdx),49); % just valence
MK =  zeros(length(trainIdx),1); % valence mean

for i=1:length(trainIdx),         
    I=find(features(:,1)==odorsID(trainIdx(i)) ); 
    K(i,features(I,2)) = features(I,4); % 3:end or 5:end
    MK(i) = mean(K(i,features(I,2)),2);
end;

LBs2 = dlmread('LBs2.txt','\t',1,2);

LBs1 = dlmread('LBs1.txt','\t',1,3);

Training_set = [molecular(trainIdx,:);molecular(leadIdx,:)];

MK = [MK;LBs2(70:138,1)];
K = [K;zeros(69,49)];
for i=1:49
    startindex = (i-1)*21*69+70;
    lastindex = startindex+68;

    K(339:end,i) = LBs1(startindex:lastindex,1);
end



Train_pred_opt = zeros(407,49);
Train_pred_M = zeros(407,1);

LB_pred_opt = zeros(69,49);
LB_pred_M = zeros(69,1);
LB_set = molecular(leadIdx,:);

FT_pred_opt = zeros(69,49);
FT_pred_M = zeros(69,1);
FT_set = molecular(testIdx,:);

%D = ones(338,1);
D = 1-mean(pdist2(FT_set,Training_set,'cosine','Smallest',5),1);
fprintf('\n\t D done');

tic;
[CoeffM,InfoM]=lasso(Training_set,MK,'CV',6,'DFmax',100,'Weights',D);
LB_pred_M = LB_set*CoeffM(:,InfoM.IndexMinMSE)+InfoM.Intercept(InfoM.IndexMinMSE);
Train_pred_M = Training_set*CoeffM(:,InfoM.IndexMinMSE)+InfoM.Intercept(InfoM.IndexMinMSE);
FT_pred_M = FT_set*CoeffM(:,InfoM.IndexMinMSE)+InfoM.Intercept(InfoM.IndexMinMSE);

%Fitting to residual
for i=1:49
    [Coeff,Info]=lasso(Training_set,K(:,i)-MK,'CV',6,'DFmax',100,'Weights',D);
    LB_pred_opt(:,i) = LB_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE)+LB_pred_M;
    Train_pred_opt(:,i) = Training_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE)+Train_pred_M;
    FT_pred_opt(:,i) = FT_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE)+FT_pred_M;
    
    fprintf('\n\t Iteration %d over',i);
end

toc

%dlmwrite('Subchallenge2LBoptPredsMean.csv',[odorsID(leadIdx) LB_pred_opt],'precision',10);
%dlmwrite('Subchallenge2LB1stdPredsMean.csv',[odorsID(leadIdx) LB_pred_1std],'precision',10);

% learn a linear model and a 2nd layer correction

P2 = [];

% now fit a cubic for each perceptual value

for prc=1:49
    prm = polyfit( Train_pred_opt(:,prc),K(:,prc),ORD);
    P2 = [P2 prm'];        
end

LB_pred_opt_adjusted = zeros(69,49);

Train_pred_opt_adjusted = zeros(407,49);

FT_pred_opt_adjusted = zeros(69,49);

for oodd = 1:length(leadIdx)
    for prc=1:49
        LB_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),LB_pred_opt(oodd,prc));
    end    
end  

for oodd = 1:length(testIdx)
    for prc=1:49
        FT_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),FT_pred_opt(oodd,prc));
    end    
end  

for oodd = 1:407
    for prc=1:49
        Train_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),Train_pred_opt(oodd,prc));
    end    
end  

%dlmwrite('Subchallenge1LBoptADJPredsV_Final.csv',[odorsID(leadIdx) LB_pred_opt_adjusted],'precision',10);
dlmwrite('Subchallenge1FToptADJPredsV_Final.csv',[odorsID(testIdx) FT_pred_opt_adjusted],'precision',10);
%dlmwrite('Subchallenge1TrainoptADJPredsV_Final.csv',[[odorsID(trainIdx);odorsID(leadIdx)] Train_pred_opt_adjusted],'precision',10);
%%
% %%%% TESTING %%%% %

% load test/leader set

% load 'leader_set.mat'; 
% added:
% leaderNum 
% leaderTxt

%load 'test_set.mat';
% added:
% testNum 
% testTxt

%% 
% preprocessing:

% I1 = find( sum(isnan( testNum(:,[1 6:27]) )') == 0 )';   
% 
% I2 = find( strcmp( testTxt( 2:end,4), 'high' ) );
% 
% goodtrials = intersect( I1, I2);
% 
% features = testNum( goodtrials, [1 6:27]);
% 
% % trials for testing Intensity
% I3 = find(strcmp(testTxt(2:end,5),'1/1,000')); 
% 
% trialsForIntens = intersect( I1, I3);
% 
% featIntens =  testNum( trialsForIntens, [1 6:7]);
% 
% % build perceptual matrix
% 
% K = zeros(length(testIdx),21); % or 19 or 21
% 
% for i=1:length(testIdx), 
%     I=find(features(:,1)==odorsID(testIdx(i)) ); 
%     % mean of features for each odor
%     K(i,:) = mean(features(I,3:end),1); % 3:end or 5:end
%     
%     % intensity:
%     I=find(featIntens(:,1)==odorsID(testIdx(i)) );    
%     K(i,1) = mean(featIntens(I,3),1);
% end;
% 
% % now predict and compare with actual
% 
% % prediction, using W and mnK learned with training set:
% x = U(testIdx,1:pcN)*W';
% y = K(:,:)- wM*repmat(mnK, [size(x,1) 1]);
% 
% % and now add 2nd layer of processing
% pred = x(:,1:end);
% pred2 = zeros(size(pred));
% 
% for oodd = 1:length(testIdx)
%     for prc=1:size(K,2)
%         pred2(oodd, prc) = polyval( P2(:,prc),pred(oodd,prc));
%     end    
% end  
% 
% act = y(:,1:end);
% 
% % evaluate
% 
% C1 = diag(corr( act' , pred2', 'type', myCorrType ));
% 
% C2 = diag(corr( act , pred2, 'type', myCorrType ));
% 
% figure, subplot(2,1,1); cdfplot(C1); title('test correlation for each odor');
% subplot(2,1,2); bar(C2); title('test correlation for each percept');

