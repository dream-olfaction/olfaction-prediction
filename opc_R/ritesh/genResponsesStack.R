#library(gam)
library(mgcv)


# read csv files data


#trainResp <- as.matrix(read.csv('trainRespFull.csv'))
#leadPred <- as.matrix(read.csv('leadPredFull_Sub.csv'))

leadPred <- as.matrix(read.csv('../data/testPredFull_Sub.csv'))
trainResp <- as.matrix(read.csv('../data/trainRespFullLead.csv'))

predictedResponses<-NULL

for(res in 1:21){

  respTrainRF100<-read.csv(file=paste0(getwd(),"/csvFinal/respTrainRF100_" ,res, ".csv"), header=TRUE) 
  respLeadRF100<-read.csv(file=paste0(getwd(),"/csvFinal/respLeadRF100_" ,res, ".csv"), header=TRUE) 
  
  respTrainRF200<-read.csv(file=paste0(getwd(),"/csvFinal/respTrainRF200_" ,res, ".csv"), header=TRUE) 
  respLeadRF200<-read.csv(file=paste0(getwd(),"/csvFinal/respLeadRF200_" ,res, ".csv"), header=TRUE) 
  
  respTrainRF300<-read.csv(file=paste0(getwd(),"/csvFinal/respTrainRF300_" ,res, ".csv"), header=TRUE) 
  respLeadRF300<-read.csv(file=paste0(getwd(),"/csvFinal/respLeadRF300_" ,res, ".csv"), header=TRUE) 
  
  respTrainRF400<-read.csv(file=paste0(getwd(),"/csvFinal/respTrainRF400_" ,res, ".csv"), header=TRUE) 
  respLeadRF400<-read.csv(file=paste0(getwd(),"/csvFinal/respLeadRF400_" ,res, ".csv"), header=TRUE) 
  
  
   respTrainRF35PC<- read.csv( file=paste0(getwd(),"/csvFinal/respTrainRF35PC" ,res, ".csv"), header=TRUE)
   respLeadRF35PC<- read.csv(file=paste0(getwd(),"/csvFinal/respLeadRF35PC" ,res, ".csv"), header=TRUE)
  
  respTrainRFLit<- read.csv( file=paste0(getwd(),"/csvFinal/respTrainRFLit_" ,res, ".csv"), header=TRUE)
  respLeadRFLit<- read.csv( file=paste0(getwd(),"/csvFinal/respLeadRFLit_" ,res, ".csv"), header=TRUE)
  
  
  
  
#respTrainGBM100<- read.csv( file=paste0(getwd(),"/csv/respTrainGBM100_" ,res, ".csv"), header=TRUE)
#respLeadGBM100<- read.csv( file=paste0(getwd(),"/csv/respLeadGBM100_" ,res, ".csv"), header=TRUE)

#respTrainGBM200<- read.csv( file=paste0(getwd(),"/csv/respTrainGBM200_" ,res, ".csv"), header=TRUE)
#respLeadGBM200<- read.csv( file=paste0(getwd(),"/csv/respLeadGBM200_" ,res, ".csv"), header=TRUE)


#respTrainGBM200FC<- read.csv( file=paste0(getwd(),"/csv/respTrainGBM200FC_" ,res, ".csv"), header=TRUE)
#respLeadGBM200FC<- read.csv( file=paste0(getwd(),"/csv/respLeadGBM200FC_" ,res, ".csv"), header=TRUE)

#respTrainGBM400FC<- read.csv( file=paste0(getwd(),"/csv/respTrainGBM400FC_" ,res, ".csv"), header=TRUE)
#respLeadGBM400FC<- read.csv( file=paste0(getwd(),"/csv/respLeadGBM400FC_" ,res, ".csv"), header=TRUE)

#respTrainGBM300FC<- read.csv( file=paste0(getwd(),"/csv/respTrainGBM300FC_" ,res, ".csv"), header=TRUE)
#respLeadGBM300FC<- read.csv( file=paste0(getwd(),"/csv/respLeadGBM300FC_" ,res, ".csv"), header=TRUE)



################# stacking of data ################################################
# 
# stackdata <-cbind(trainResp[,res],respTrainRF100,respTrainRF200,respTrainRF35PC, respTrainRFLit)
# colnames(stackdata)<-c('TruePred', 'RF100','RF200','RF35PC','RFLit')

#stackdata <-cbind(trainResp[,res],respTrainRF100,respTrainRF200, respTrainRF35PC,respTrainRFLit, respTrainGBM100, respTrainGBM200, respTrainGBM200FC)
#colnames(stackdata)<-c('TruePred', 'RF100','RF200','RF35PC','RFLit','RFGBM100','RFGBM200','RFGBM200FC')
#stackdata<-(stackdata)

stackdata <-cbind(trainResp[,res],respTrainRF100,respTrainRF200, respTrainRF300, respTrainRF400, respTrainRF35PC,respTrainRFLit)
colnames(stackdata)<-c('TruePred', 'RF100','RF200','RF300', 'RF400','RF35PC','RFLit')
stackdata<-(stackdata)



# stacktest <-cbind(respLeadRF100,respLeadRF200,respLeadRF35PC,respLeadRFLit )
# colnames(stacktest)<-c('RF100','RF200','RF35PC','RFLit')    

stacktest <-cbind(respLeadRF100,respLeadRF200,respLeadRF300,respLeadRF400,respLeadRF35PC,respLeadRFLit )
colnames(stacktest)<-c('RF100','RF200','RF300','RF400', 'RF35PC','RFLit')    

stacktest<-(stacktest)        

# stacking of all models
# family= ziP(theta = NULL, link = "identity")

gam.mod<-gam( TruePred ~ s(RF100) + s(RF200) +  s(RF300) +  s(RF400) +s(RF35PC) +  s(RFLit), 
            family= ziP(theta = NULL, link = "identity") , data = stackdata, trace=TRUE)

save(gam.mod, file = paste0(getwd(),"/modelsFinal/gammodnew_" ,res, ".RData"), compress= TRUE)

########### final predicted response , write it in file###########
temp <- predict(gam.mod, newdata=stacktest , type="response")
predictedResponses<-cbind(predictedResponses, temp)

}

predictedResponses<-cbind(leadPred[,c(1,2)], predictedResponses)

write.csv( predictedResponses, file=paste0(getwd(),"/csvFinal/ResponsesFinalFunc1_21.csv"), row.names = FALSE)

