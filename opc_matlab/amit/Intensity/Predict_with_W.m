
clear

load '..\train_set.mat';

W = csvread('BestW_gammaStdFinal_no_cid.csv');

molecular = MolecularNum;

[i j] = find( isnan(molecular) );

molecular(i,j)=0;

%[U S V] = svd(log(100+molecular),'econ');

molecular(:,2:end) = log(100+molecular(:,2:end));


Inputs = molecular;
Targets = csvread('Subchallenge1TrainTargetInt.csv');

Targets = [Targets;zeros(69,49)];

LBs1 = dlmread('LBs1.txt','\t',1,3);

for i=1:49
    startindex = (i-1)*21*69+1;
    lastindex = startindex+68;

    Targets(339:end,i) = LBs1(startindex:lastindex,1);
end

predictions_LB = zeros(69,49);

predictions_FT = zeros(69,49);

predictions_TR = zeros(407,49);

Training_set = [Inputs(trainIdx,2:end);Inputs(leadIdx,2:end)];

LB_set = Inputs(leadIdx,2:end);

FT_set = Inputs(testIdx,2:end);


for i=1:49
    predictions_TR(:,i) = [Training_set]*W(:,i);
    predictions_LB(:,i) = [LB_set]*W(:,i);
    predictions_FT(:,i) = [FT_set]*W(:,i);
end


P2 = [];

% now fit a cubic for each perceptual value

for prc=1:49
    prm = polyfit( predictions_TR(:,prc),Targets(:,prc),3);
    P2 = [P2 prm'];        
end

LB_pred_opt_adjusted = zeros(69,49);
FT_pred_opt_adjusted = zeros(69,49);
Train_pred_opt_adjusted = zeros(407,49);
%pred = firstLayer(:,1:end);
%pred2 = zeros(size(pred));

for oodd = 1:length(leadIdx)
    for prc=1:49
        LB_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),predictions_LB(oodd,prc));
        FT_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),predictions_FT(oodd,prc));
    end    
end  

for oodd = 1:length(trainIdx)
    for prc=1:49
        Train_pred_opt_adjusted(oodd, prc) = polyval( P2(:,prc),predictions_TR(oodd,prc));
    end    
end  

%dlmwrite('Subchallenge1LBoptmtflADJPredsStdFinal_no_cid.csv',[odorsID(leadIdx) LB_pred_opt_adjusted],'precision',10);
dlmwrite('Subchallenge1FToptmtflADJPredsStdFinal_no_cid.csv',[odorsID(testIdx) FT_pred_opt_adjusted],'precision',10);
%dlmwrite('Subchallenge1TrainoptmtflADJPredsStdFinal_no_cid.csv',[[odorsID(trainIdx);odorsID(leadIdx)] Train_pred_opt_adjusted],'precision',10);


