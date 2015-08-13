library(randomForest)
library(foreach)
library(doMC)

#setwd('D:/PhD/olfaction challenge/r_olfaction-2015-03-22/r olfaction')

registerDoMC(cores=4)

#trainResp <- as.matrix(read.csv('trainRespFull.csv'))

#trainPredPC <- as.matrix(read.csv('trainPredPC.csv'))

#trainPred <- as.matrix(read.csv('trainPredFull.csv'))


trainPred <- as.matrix(read.csv('/../data/trainPredFullLead.csv'))
trainPredPC <- as.matrix(read.csv('../data/trainPredPC40Lead.csv'))
trainResp <- as.matrix(read.csv('../data/trainRespFullLead.csv'))




# For all the responses 2:21

for(res in 2:21)
{
  rfPC.mod <- foreach(ntree=rep(250, 4), .combine = combine, .multicombine = TRUE,
                    .packages = 'randomForest') %dopar% {
                      randomForest(trainPredPC[,c(2:ncol(trainPredPC))], trainResp[,res], ntree=ntree)
                    }
  rf.mod <- foreach(ntree=rep(300, 4), .combine = combine, .multicombine = TRUE,
                    .packages = 'randomForest') %dopar% {
                      randomForest(trainPred[,c(2:ncol(trainPred))], trainResp[,res], ntree=ntree)
                    }
  save(rfPC.mod, file = paste0(getwd(),"/RDataFinal/rfmodPC" , res , ".RData"), compress= TRUE)
  save(rf.mod, file = paste0(getwd(),"/RDataFinal/rfmod" , res , ".RData"), compress= TRUE)
  
  
  
}
