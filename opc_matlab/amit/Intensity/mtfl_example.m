%trainx, testx: feature matrix -- each column is an example (num_features x num_examples_for_all_tasks)
%trainy, testy: labels for all examples -- length: num_examples_for_all_tasks
%task_indices: vector of lenght T (number of tasks) -- i'th element is the starting index of examples for i'th task in trainx and trainy
%task_indices_test: same as 'task_indices' for test data 


%best mean corr is gamma = 0.3, best std corr is gamma = 5 
clear

load 'train_set.mat';

T = 49;
dim = 4869;
cv_size = 6;
Dini = eye(dim)/dim;

iterations = 10;
method_str = 'feat';
epsilon_init = 0;
fname = 'school';
gammas = [0.3,5];

molecular = MolecularNum;

[i j] = find( isnan(molecular) );

molecular(i,j)=0;

%[U S V] = svd(log(100+molecular),'econ');

molecular(:,2:end) = log(100+molecular(:,2:end));

Targets = csvread('Subchallenge1TrainTargetInt.csv');

Targets = [Targets;zeros(69,49)];

LBs1 = dlmread('LBs1.txt','\t',1,3);

for i=1:49
    startindex = (i-1)*21*69+1;
    lastindex = startindex+68;

    Targets(339:end,i) = LBs1(startindex:lastindex,1);
end

Training_set = [];

LB_set = [];

FT_set = [];


target_length = T*407;

Training_Target = zeros(target_length,1);

test_length = T*69;

Test_Target = zeros(test_length,1);

task_indices = zeros(1,T);
task_indices_test = zeros(1,T);

for i=1:T
    indexst = (i-1)*407+1;
    indexend = i*407;
    Training_Target(indexst:indexend) = Targets(:,i);
    Training_set = [Training_set,[molecular(trainIdx,2:end);molecular(leadIdx,2:end)]'];
    %LB_set = [LB_set,[molecular(leadIdx,2:end)]'];
    FT_set = [FT_set,[molecular(testIdx,2:end)]'];
    task_indices(i) = indexst;
    task_indices_test(i) = (i-1)*69+1;
end

fprintf('\n\t Input ready');

for i=1:2
    
tic;
[bestcv,bestgamma,cverrs,testerrs,theW,theD] = ...
run_code_example(gammas(i),Training_set,Training_Target,FT_set,Test_Target,task_indices,task_indices_test,cv_size,Dini,iterations,method_str, epsilon_init, fname);

toc
gammas
    if i==1
        dlmwrite('BestW_gammaMeanFinal_no_cid.csv',theW,'precision',10);
    elseif i==2
         dlmwrite('BestW_gammaStdFinal_no_cid.csv',theW,'precision',10);   
    end
 end