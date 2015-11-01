# Best parameters
#
#use_et: Whether to use ExtraTreesRegressor instead of RandomForestRegressor
#max_depth: 2,6,15,32,None
#min_samples: 1,4,16,64
#regularize(cols 0-20):
#transform_weight(cols 21-41): For stdevs, how much weight to give the mean transform
#transform_params: For stdevs, what coefficients to use in the mean transform
#use_mask: Whether to use the masked data or the imputed data
#n_estimators=25 for all params derived here
#
#Format:
#col:[use_et,max_features,max_depth,min_samples,regularize_or_transform_weight,use_mask] 
best = {col:{} for col in range(42)}
best[0]=[True,None,None,1,0.8,False]          
best[1]=[False,None,None,1,0.7,False]         
best[2]=[False,None,6,1,0.9,False]      
best[3]=[False,None,None,1,0.8,False]                
best[4]=[False,None,15,1,0.8,False]                        
best[5]=[False,200,32,4,0.95,False]      
best[6]=[False,200,15,1,0.8,False]      
best[7]=[False,200,6,4,0.8,False]      
best[8]=[False,200,15,1,0.9,False]      
best[9]=[False,200,15,1,0.8,False]      
best[10]=[False,None,15,1,0.85,False]                       
best[11]=[False,None,15,1,0.9,False]                       
best[12]=[False,None,6,1,0.9,False]                        
best[13]=[False,None,6,1,0.75,False]                        
best[14]=[False,200,15,1,0.85,False]      
best[15]=[False,None,15,1,0.75,False]                            
best[16]=[False,200,15,1,0.8,False]      
best[17]=[False,200,2,4,0.85,False]      
best[18]=[False,200,15,1,0.9,False]      
best[19]=[False,200,6,1,0.8,False]      
best[20]=[False,None,32,1,0.9,False]      
best[21]=[False,200,2,64,1.0,True]
best[22]=[False,200,None,2,0.03,True]
best[23]=[False,200,6,1,1.0,True]
best[24]=[False,200,6,4,0.62,True]
best[25]=[False,None,6,1,0.79,True]            
best[26]=[False,200,6,16,0.97,True]
best[27]=[False,200,6,4,0.68,True]
best[28]=[False,None,6,1,0.88 ,True]            
best[29]=[False,200,6,16,1.0,True]
best[30]=[False,200,None,1,0.76,True]
best[31]=[False,200,6,4,0.91,True]
best[32]=[False,None,2,1,0.5,True]              
best[33]=[False,200,32,1,1.0,True]
best[34]=[False,200,6,4,0.35,True]
best[35]=[False,200,32,1,0.67,True]
best[36]=[False,200,2,4,1.0,True]
best[37]=[False,200,6,4,1.0,True]
best[38]=[False,200,2,1,1.0,True]
best[39]=[False,200,6,1,1.0,True]
best[40]=[False,200,6,4,1.0,True]
best[41]=[False,None,15,1,0.38,True]              