library(randomForest)
library(foreach) # for parallel
library(doMC) # for multiple cores




# read csv files data
#trainPred <- as.matrix(read.csv('trainPredFull.csv')) # this file has 3084 rows...

trainPred <- as.matrix(read.csv('../data/trainPredFullLead.csv')) # this file has 3084 + lead molecules rows...

#leadPred <- as.matrix(read.csv('leadPredFull_Sub.csv'))
# for testPred read the csv file testPredFull_Sub.csv

leadPred <- as.matrix(read.csv('../data/testPredFull_Sub.csv'))


#trainResp <- as.matrix(read.csv('trainRespFull.csv'))

trainResp <- as.matrix(read.csv('../data/trainRespFullLead.csv'))
                                                                                                                                                                                                                                                                                                                                                                                                                                


#trainPredPC <- as.matrix(read.csv('trainPredPC.csv'))

trainPredPC <- as.matrix(read.csv('../data/trainPredPC40Lead.csv'))

#leadPredPC <- as.matrix(read.csv('leadPred.csv'))

leadPredPC <- as.matrix(read.csv('../data/testPredPC.csv'))

########################### Register number of cores on CPU to be used for this task ##################################

 registerDoMC(cores=4) # Ritesh : It works on linux 
######################################################################################################

# for all responses 1:21
for(res in 1: 21)
{ # start of for loop for variable
  # save models, training and prediction on trainPred......
  # at the end will predict for leaderPred
  #load random forest model, variable > rf.mod
  
  load (paste0(getwd(),"/RDataFinal/rfmod",  res,".RData") )
  
  load(paste0(getwd(),"/RDataFinal/rfmodPC",  res,".RData"))
  
  # Model A : train RF model with 100 features.
  # select  first 100 features from rf.mod
  
  imp <- as.data.frame(rf.mod$importance[order(rf.mod$importance,decreasing = TRUE),])
  
  # select 100 features from it 
  Selrow<-rownames(imp)[1:100]
  ########################## see the for each part for parallel calculations ################################
  
  RF100.mod <- foreach(ntree=rep(250, 4), .combine=combine, .multicombine=TRUE,
                       .packages='randomForest') %dopar% {
                         randomForest(trainPred[,Selrow], trainResp[,res], ntree=ntree)
                       }
  save(RF100.mod, file = paste0(getwd(),"/modelsFinal/RF100mod_" ,res, ".RData"), compress= TRUE)
  
  respTrainRF100<- predict(RF100.mod, trainPred[,Selrow], type="response",
                           predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  respLeadRF100<- predict(RF100.mod, leadPred[,Selrow], type="response",
                          predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  
  write.csv(respTrainRF100, file=paste0(getwd(),"/csvFinal/respTrainRF100_" ,res, ".csv"), row.names = FALSE)
  write.csv( respLeadRF100, file=paste0(getwd(),"/csvFinal/respLeadRF100_" ,res, ".csv"), row.names = FALSE)
  
  
  # Model B : train RF model with 200 features.   
  
  Selrow<-rownames(imp)[1:200]
  RF200.mod <- foreach(ntree=rep(250, 4), .combine=combine, .multicombine=TRUE,
                       .packages='randomForest') %dopar% {
                         randomForest(trainPred[,Selrow], trainResp[,res], ntree=ntree)
                       }
  save(RF200.mod, file = paste0(getwd(),"/modelsFinal/RF200mod_" ,res, ".RData"), compress= TRUE)
  
  respTrainRF200<- predict(RF200.mod, trainPred[,Selrow], type="response",
                           predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  respLeadRF200<- predict(RF200.mod, leadPred[,Selrow], type="response",
                          predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  
  write.csv(respTrainRF200, file=paste0(getwd(),"/csvFinal/respTrainRF200_" ,res, ".csv"), row.names = FALSE)
  write.csv( respLeadRF200, file=paste0(getwd(),"/csvFinal/respLeadRF200_" ,res, ".csv"), row.names = FALSE)
  
  # Model C : train RF model with 300 features.   
  
  Selrow<-rownames(imp)[1:300]
  RF300.mod <- foreach(ntree=rep(300, 4), .combine=combine, .multicombine=TRUE,
                       .packages='randomForest') %dopar% {
                         randomForest(trainPred[,Selrow], trainResp[,res], ntree=ntree)
                       }
  save(RF300.mod, file = paste0(getwd(),"/modelsFinal/RF300mod_" ,res, ".RData"), compress= TRUE)
  
  respTrainRF300<- predict(RF300.mod, trainPred[,Selrow], type="response",
                           predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  respLeadRF300<- predict(RF300.mod, leadPred[,Selrow], type="response",
                          predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  
  write.csv(respTrainRF300, file=paste0(getwd(),"/csvFinal/respTrainRF300_" ,res, ".csv"), row.names = FALSE)
  write.csv( respLeadRF300, file=paste0(getwd(),"/csvFinal/respLeadRF300_" ,res, ".csv"), row.names = FALSE)
  
  
  # Model D : train RF model with 400 features.   
  
  Selrow<-rownames(imp)[1:400]
  RF400.mod <- foreach(ntree=rep(350, 4), .combine=combine, .multicombine=TRUE,
                       .packages='randomForest') %dopar% {
                         randomForest(trainPred[,Selrow], trainResp[,res], ntree=ntree)
                       }
  save(RF400.mod, file = paste0(getwd(),"/modelsFinal/RF400mod_" ,res, ".RData"), compress= TRUE)
  
  respTrainRF400<- predict(RF300.mod, trainPred[,Selrow], type="response",
                           predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  respLeadRF400<- predict(RF300.mod, leadPred[,Selrow], type="response",
                          predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  
  write.csv(respTrainRF400, file=paste0(getwd(),"/csvFinal/respTrainRF400_" ,res, ".csv"), row.names = FALSE)
  write.csv(respLeadRF400, file=paste0(getwd(),"/csvFinal/respLeadRF400_" ,res, ".csv"), row.names = FALSE)
  
  
#   # Model E : train RF model with PC features.   
#   
#   
   imp <- as.data.frame(rfPC.mod$importance[order(rfPC.mod$importance,decreasing = TRUE),])
#   
#   # Seelct some featrures
#   
   Selrow<-rownames(imp)[1:35]
   RF35PC.mod <- foreach(ntree=rep(250, 4), .combine=combine, .multicombine=TRUE,
                        .packages='randomForest') %dopar% {
                         randomForest(trainPredPC[,Selrow], trainResp[,res], ntree=ntree)
                        }
   save(RF35PC.mod, file = paste0(getwd(),"/models/RF35PC" ,res, ".RData"), compress= TRUE)
   
   respTrainRF35PC<- predict(RF35PC.mod, trainPredPC[,Selrow], type="response",
                            predict.all=FALSE, proximity=FALSE, nodes=FALSE)
   respLeadRF35PC<- predict(RF35PC.mod, leadPredPC[,Selrow], type="response",
                           predict.all=FALSE, proximity=FALSE, nodes=FALSE)
   
   write.csv(respTrainRF35PC, file=paste0(getwd(),"/csvFinal/respTrainRF35PC" ,res, ".csv"), row.names = FALSE)
   write.csv(respLeadRF35PC, file=paste0(getwd(),"/csvFinal/respLeadRF35PC" ,res, ".csv"), row.names = FALSE)
   
  
  
  #Model F: features from literature, ritesh you provided 34 features but in trainPred, these 9 of these features are not present  
  
#   FeatLit<- c( 'MW',  'Sv',  'Se',	'nDB',	'nBnz',	'MSD',	'MAXDN',	'GATS1m',	'JGT',	'SpMin8_Bh(m)'	, 'Eig12_EA(ed)',	'Eig11_EA(ri)',
#                'SPAM',	'AROM',	'DISPm',	'DISPe'	,'Mor23m',	'Mor25m',	'Mor04v',	'Mor08v'	,'Mor29e',	'P2m',	'G1m',	'HATS6m',	'R2m+'	,'R1e+',	'R2p'	,'nCp',
#                'nRCOOH',	'nArOH',	'C-027',	'C-040',	'O-057',	'N-075')


  FeatLit<- c( 'MW',  'Sv',  'Se',  'nDB',	'nBnz',	'MSD',	'MAXDN',	'GATS1m',	'JGT',	
             'SPAM',	'AROM',	'DISPm',	'DISPe'	,'Mor23m',	'Mor25m',	'Mor04v',	'Mor08v'	,'Mor29e',	'P2m',	'G1m',	'HATS6m',		'R2p'	,'nCp',
             'nRCOOH',	'nArOH')




  
  RFLit.mod <- foreach(ntree=rep(250, 4), .combine=combine, .multicombine=TRUE,
                       .packages='randomForest') %dopar% {
                         randomForest(trainPred[,FeatLit], trainResp[,res], ntree=ntree)
                       }
  save(RFLit.mod, file = paste0(getwd() , "/modelsFinal/RFLit_" ,res, ".RData"), compress= TRUE)
  
  respTrainRFLit<- predict(RFLit.mod, trainPred[,FeatLit], type="response", predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  respLeadRFLit<- predict(RFLit.mod, leadPred[,FeatLit], type="response",   predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  
  write.csv(respTrainRFLit, file=paste0(getwd(), "/csvFinal/respTrainRFLit_" ,res, ".csv"), row.names = FALSE)
  write.csv( respLeadRFLit, file=paste0(getwd(), "/csvFinal/respLeadRFLit_" ,res, ".csv"), row.names = FALSE)
  
    



  
}# end of variable for loop

