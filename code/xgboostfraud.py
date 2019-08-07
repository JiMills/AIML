
import pandas as pd
import numpy as np
import gc
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import xgboost as xgb
import os

#TRAIN/VALIDATION/TEST SPLIT
#VALIDATION
VALID_SIZE = 0.20 # simple validation using train_test_split
TEST_SIZE = 0.20 # test size using_train_test_split
RANDOM_STATE = 2018
MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 200 #lgb early stop
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result

MyProjectRepo = os.popen('bdvcli --get cluster.project_repo').read().rstrip()

print (MyProjectRepo)
# should input from command line the source of csv file

# could be github or repo or somewhere else

print ("\nLoading Credit Card Data from: " + MyProjectRepo + "/data/fraud/creditcard.csv" )
data_df = pd.read_csv(MyProjectRepo + "/data/fraud/creditcard.csv")
print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])

print ("\n")

#Predictive Model

target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']

train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )
dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

#Train with xgboost GPU

params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.039
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = RANDOM_STATE
#params['gpu_id'] = 0
#params['tree_method'] = 'gpu_exact'
# gpu_hist causes a memory error
params['tree_method'] = 'gpu_hist'

print ("\nTraining with GPU\n")

GPUstartTime = datetime.now()
GPUmodel = xgb.train(params,
                dtrain,
                MAX_ROUNDS,
                watchlist,
                early_stopping_rounds=EARLY_STOP,
                maximize=True,
                verbose_eval=VERBOSE_EVAL)

GPUendTime = datetime.now()


#Train with xgboost CPU

params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.039
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = RANDOM_STATE

print ("\nPausing for 30 seconds")
time.sleep(30)

print ("Traning with CPU\n")

CPUstartTime = datetime.now()
CPUmodel = xgb.train(params,
                dtrain,
                MAX_ROUNDS,
                watchlist,
                early_stopping_rounds=EARLY_STOP,
                maximize=True,
                verbose_eval=VERBOSE_EVAL)

CPUendTime = datetime.now()

print("\nCredit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])
print("\nGPU Traning Time:", GPUendTime - GPUstartTime)

# Display accuracy
preds = GPUmodel.predict(dtest)
print ("GPU Accuracy: %.2f%%"  % (roc_auc_score(test_df[target].values, preds)*100.0))

print("\nCPU Traing Time:", CPUendTime - CPUstartTime)

# Display accuracy
preds = CPUmodel.predict(dtest)
print ("CPU Accuracy: %.2f%%"  % (roc_auc_score(test_df[target].values, preds)*100.0))
print ("\n")

# save model to file
# this should be in /model/
joblib.dump(GPUmodel,MyProjectRepo +  "/GPU_credit_fraud_xgboost.model")
joblib.dump(CPUmodel, MyProjectRepo + "/CPU_credit_fraud_xgboost.model")

