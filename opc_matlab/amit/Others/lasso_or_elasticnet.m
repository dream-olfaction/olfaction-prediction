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

runs = 21;
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
%delta = 0.001;
%minval = min(min(molecular(:,2:end))')

molecular(:,2:end) = log(100+molecular(:,2:end));

%molecular = [molecular molecular(:,2:end).^2];

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
    K(i,:) = mean(features(I,3:end),1); % 3:end or 5:end
    
    % intensity:
    I=find(featIntens(:,1)==odorsID(trainIdx(i)) );    
    K(i,1) = mean(featIntens(I,3),1);
end;

% intensity for odors without 1/1,000
for i=1:length(noOneThou)
    % I1: all good trials
    I4 = find( trainNum(:,1) == noOneThou(i) );
    I5 = intersect( I1, I4 );
    K( noOT_list(i), 1 ) = mean( trainNum(I5,7) );
end

mnK = mean(K);
K = K - wM*repmat( mnK, [size(K,1) 1]);% Prediction Matrix for mean ready

%csvwrite('Subchallenge2TrainTargetMean.csv',K);
LBs2 = dlmread('LBs2.txt','\t',1,2);

Training_set = [molecular(trainIdx,:);molecular(leadIdx,:)];

K = [K;zeros(69,21)];
for i=1:21
    startindex = (i-1)*69+1;
    lastindex = i*69;

    K(339:end,i) = LBs2(startindex:lastindex,1);
end



Train_pred_opt = zeros(407,21);


LB_pred_opt = zeros(69,21);
%LB_pred_1std = zeros(69,21);
LB_set = molecular(leadIdx,:);

FT_pred_opt = zeros(69,21);
%FT_pred_1std = zeros(69,21);
FT_set = molecular(testIdx,:);

D = ones(407,1);
D = 1-mean(pdist2(FT_set,Training_set,'cosine','Smallest',5),1);
fprintf('\n\t D done');
tic;
for i=3:21
    [Coeff,Info]=lasso(Training_set,K(:,i),'CV',6,'DFmax',5000,'Alpha',0.95,'Weights',D);
    LB_pred_opt(:,i) = LB_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE);
    Train_pred_opt(:,i) = Training_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE);
    FT_pred_opt(:,i) = FT_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE);
    %LB_pred_1std(:,i) = LB_set*Coeff(:,Info.Index1SE)+Info.Intercept(Info.Index1SE);
    %FT_pred_1std(:,i) = FT_set*Coeff(:,Info.Index1SE)+Info.Intercept(Info.Index1SE);
    fprintf('\n\t Iteration %d over',i);
end
toc

%dlmwrite('Subchallenge2LBoptPredsMean.csv',[odorsID(leadIdx) LB_pred_opt],'precision',10);
%dlmwrite('Subchallenge2LB1stdPredsMean.csv',[odorsID(leadIdx) LB_pred_1std],'precision',10);

% learn a linear model and a 2nd layer correction

P2 = [];

% now fit a cubic for each perceptual value

for prc=1:runs
    prm = polyfit( Train_pred_opt(:,prc),K(:,prc),ORD);
    P2 = [P2 prm'];        
end

LB_pred_opt_adjusted = zeros(69,21);

Train_pred_opt_adjusted = zeros(338,21);

FT_pred_opt_adjusted = zeros(69,21);

for oodd = 1:length(leadIdx)
    for prc=1:runs
        LB_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),LB_pred_opt(oodd,prc));
    end    
end  

for oodd = 1:length(testIdx)
    for prc=1:runs
        FT_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),FT_pred_opt(oodd,prc));
    end    
end  

for oodd = 1:length(trainIdx)
    for prc=1:runs
        Train_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),Train_pred_opt(oodd,prc));
    end    
end  

% dlmwrite('Subchallenge2LBoptlas6foldADJPredsMeanV_Final.csv',[odorsID(leadIdx) LB_pred_opt_adjusted],'precision',10);
% dlmwrite('Subchallenge2FToptlas6foldADJPredsMeanV_Final.csv',[odorsID(testIdx) FT_pred_opt_adjusted],'precision',10);
% dlmwrite('Subchallenge2Trainoptlas6foldADJPredsMeanV_Final.csv',[odorsID(trainIdx) Train_pred_opt_adjusted],'precision',10);


%dlmwrite('Subchallenge2LBelas0956foldADJPredsMean_cos_OthersFinal.csv',[odorsID(leadIdx) LB_pred_opt_adjusted],'precision',10);
dlmwrite('Subchallenge2FTelas0956foldADJPredsMean_cos_OthersFinal.csv',[odorsID(testIdx) FT_pred_opt_adjusted],'precision',10);
%dlmwrite('Subchallenge2Trainelas0956foldADJPredsMean_cos_OthersFinal.csv',[odorsID(trainIdx) Train_pred_opt_adjusted],'precision',10);
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

