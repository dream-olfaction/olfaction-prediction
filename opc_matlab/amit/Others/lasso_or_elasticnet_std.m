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

LB = dlmread('LBs2.txt','\t',1,2);

%%
% reduce molecular descriptors' dimensionality

molecular = MolecularNum;

[i j] = find( isnan(molecular) );

molecular(i,j)=0;


%delta = 0.001;
%minval = min(min(molecular(:,2:end))')

molecular(:,2:end) = log(100+molecular(:,2:end));

% [U S V] = svd(molecular(:,2:end),'econ');
% 
% [Utrrnd Strrnd V] = svd(molecular(trainIdx(randi(338,69,1)),2:end),'econ');
% % 
% [Utr1 Str1 V] = svd(molecular(trainIdx(1:69),2:end),'econ');
% [Utr2 Str2 V] = svd(molecular(trainIdx(70:138),2:end),'econ');
% % 
% [Utr3 Str3 V] = svd(molecular(trainIdx(139:207),2:end),'econ');
% % 
% [Ulb Slb V] = svd(molecular(leadIdx,2:end),'econ');
% % 
%  [Utst Stst V] = svd(molecular(testIdx,2:end),'econ');
% 
%  angle_lb_tst = 0;
%  angle_rndtr_tst = 0;
%  angle_seqtr1_tst = 0;
%  angle_seqtr2_tst = 0;
%  angle_seqtr3_tst = 0;
%  angle_tst_tst = 0;
%  angle_lb_rndtr = 0;
%  angle_lb_seqtr1 = 0;
%  for i=1:69
%      angle_lb_tst = angle_lb_tst + 1 - abs(dot(Ulb(:,i),Utst(:,i))/(sqrt(dot(Ulb(:,i),Ulb(:,i)))*sqrt(dot(Utst(:,i),Utst(:,i)))));
%      angle_rndtr_tst = angle_rndtr_tst + 1 - abs(dot(Utrrnd(:,i),Utst(:,i))/(sqrt(dot(Utrrnd(:,i),Utrrnd(:,i)))*sqrt(dot(Utst(:,i),Utst(:,i)))));
%      angle_seqtr1_tst = angle_seqtr1_tst + 1 - abs(dot(Utr1(:,i),Utst(:,i))/(sqrt(dot(Utr1(:,i),Utr1(:,i)))*sqrt(dot(Utst(:,i),Utst(:,i)))));
%      angle_seqtr2_tst = angle_seqtr2_tst + 1 - abs(dot(Utr2(:,i),Utst(:,i))/(sqrt(dot(Utr2(:,i),Utr2(:,i)))*sqrt(dot(Utst(:,i),Utst(:,i)))));
%      angle_seqtr3_tst = angle_seqtr3_tst + 1 - abs(dot(Utr3(:,i),Utst(:,i))/(sqrt(dot(Utr3(:,i),Utr3(:,i)))*sqrt(dot(Utst(:,i),Utst(:,i)))));
%      angle_tst_tst = angle_tst_tst + 1 - abs(dot(Utst(:,i),Utst(:,i))/(sqrt(dot(Utst(:,i),Utst(:,i)))*sqrt(dot(Utst(:,i),Utst(:,i)))));
%      angle_lb_rndtr = angle_lb_rndtr + 1 - abs(dot(Ulb(:,i),Utrrnd(:,i))/(sqrt(dot(Ulb(:,i),Ulb(:,i)))*sqrt(dot(Utrrnd(:,i),Utrrnd(:,i)))));
%      angle_lb_seqtr1 = angle_lb_seqtr1 + 1 - abs(dot(Ulb(:,i),Utr1(:,i))/(sqrt(dot(Ulb(:,i),Ulb(:,i)))*sqrt(dot(Utr1(:,i),Utr1(:,i)))));
%      
%  end
%  
%  angle_tst_tst = angle_tst_tst/69
%  angle_lb_tst = angle_lb_tst/69
%  angle_rndtr_tst = angle_rndtr_tst/69
%  angle_seqtr1_tst = angle_seqtr1_tst/69
%  angle_seqtr2_tst = angle_seqtr2_tst/69
%  angle_seqtr3_tst = angle_seqtr3_tst/69
%  angle_lb_rndtr = angle_lb_rndtr/69
%  angle_lb_seqtr1 = angle_lb_seqtr1/69
 
%molecular = [molecular molecular(:,2:end).^2 molecular(:,2:end).^3];

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

% learn a linear model and a 2nd layer correction

mnK = mean(K);
K = K - wM*repmat( mnK, [size(K,1) 1]);

%csvwrite('Subchallenge2TrainTargetStd.csv',K);

scale_factor = 1;

K = scale_factor*K;

LBs2 = dlmread('LBs2.txt','\t',1,2);

Training_set = [molecular(trainIdx,:);molecular(leadIdx,:)];

K = [K;zeros(69,21)];
for i=1:21
    startindex = (i-1)*69+1;
    lastindex = i*69;

    K(339:end,i) = LBs2(startindex:lastindex,2);
end

Train_pred_opt = zeros(407,21);

LB_pred_opt = zeros(69,21);
%LB_pred_1std = zeros(69,21);
LB_set = molecular(leadIdx,:);

FT_pred_opt = zeros(69,21);
%FT_pred_1std = zeros(69,21);
FT_set = molecular(testIdx,:);

D = ones(407,1);
%D = 1-mean(pdist2(LB_set,Training_set,'cosine','Smallest',5),1);
fprintf('\n\t D done');



tic;
for i=3:runs
   [Coeff,Info]=lasso(Training_set,K(:,i),'CV',6,'DFmax',100,'Alpha',1,'Weights',D);
    LB_pred_opt(:,i) = LB_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE);
    Train_pred_opt(:,i) = Training_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE);
    FT_pred_opt(:,i) = FT_set*Coeff(:,Info.IndexMinMSE)+Info.Intercept(Info.IndexMinMSE);
    %LB_pred_1std(:,i) = LB_set*Coeff(:,Info.Index1SE)+Info.Intercept(Info.Index1SE);
    %FT_pred_1std(:,i) = FT_set*Coeff(:,Info.Index1SE)+Info.Intercept(Info.Index1SE);
    fprintf('\n\t Iteration %d over',i);
end
toc

%dlmwrite('Subchallenge2LBoptPredsStd.csv',[odorsID(leadIdx) LB_pred_opt],'precision',10);
%dlmwrite('Subchallenge2LB1stdPredsStd.csv',[odorsID(leadIdx) LB_pred_1std],'precision',10);

% learn a linear model and a 2nd layer correction


P2 = [];

% now fit a cubic for each perceptual value

for prc=1:runs
    prm = polyfit( Train_pred_opt(:,prc),K(:,prc),ORD);
    P2 = [P2 prm'];        
end

LB_pred_opt_adjusted = zeros(69,21);

Train_pred_opt_adjusted = zeros(407,21);

FT_pred_opt_adjusted = zeros(69,21);

for oodd = 1:length(leadIdx)
    for prc=1:runs
        LB_pred_opt_adjusted(oodd, prc) = (polyval( P2(:,prc),LB_pred_opt(oodd,prc)))/scale_factor;
    end    
end  

for oodd = 1:length(testIdx)
    for prc=1:runs
        FT_pred_opt_adjusted(oodd, prc) = (polyval( P2(:,prc),FT_pred_opt(oodd,prc)))/scale_factor;
    end    
end  

for oodd = 1:407
    for prc=1:runs
        Train_pred_opt_adjusted(oodd, prc) = (polyval( P2(:,prc),Train_pred_opt(oodd,prc)))/scale_factor;
    end    
end  

%dlmwrite('Subchallenge2LBoptlas6foldADJPredsStd_OthersFinal.csv',[odorsID(leadIdx) LB_pred_opt_adjusted],'precision',10);
dlmwrite('Subchallenge2FToptlas6foldADJPredsStd_OthersFinal.csv',[odorsID(testIdx) FT_pred_opt_adjusted],'precision',10);
%dlmwrite('Subchallenge2Trainoptlas6foldADJPredsStd_OthersFinal.csv',[[odorsID(trainIdx);odorsID(leadIdx)] Train_pred_opt_adjusted],'precision',10);

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

