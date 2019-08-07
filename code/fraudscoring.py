#!/usr/bin/python3

# last updated  7/26/19
# DISCLAIMER OF WARRANTY
#This document may contain the following HPE or other software: XML, CLI statements, scripts, parameter files. These are provided as a courtesy,
#free of charge, AS-IS by Hewlett-Packard Enterprise Company (HPE). HPE shall have no obligation to maintain or support this software. HPE MAKES
#NO EXPRESS OR IMPLIED WARRANTY OF ANY KIND REGARDING THIS SOFTWARE INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
#TITLE OR NON-INFRINGEMENT. HPE SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES, WHETHER BASED ON CONTRACT,
#TORT OR ANY OTHER LEGAL THEORY, CONNECTION WITH OR ARISING OUT OF THE FURNISHING, PERFORMANCE OR USE OF THIS SOFTWARE.

#Copyright 2011 Hewlett-Packard Enterprise Development Company, L.P. The information contained herein is subject to change without notice. The only
#warranties for HPE products and services are set forth in the express warranty statements accompanying such products and services. Nothing
#herein should be construed as constituting an additional warranty. HPE shall not be liable for technical or editorial errors or omissions
#contained herein.

import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.externals import joblib
import xgboost as xgb
import os


MyProjectRepo = os.popen('bdvcli --get cluster.project_repo').read().rstrip()

print (MyProjectRepo)

# could be github or repo or somewhere else

target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']

#init the model
load_GPUmodel = xgb.Booster()
# load model from file
load_CPUmodel = joblib.load(MyProjectRepo +  "/GPU_credit_fraud_xgboost.model")

cli_input = sys.argv[1]

data_df = pd.read_csv(MyProjectRepo + "mytestf.csv")
testdf = xgb.DMatrix(data_df[predictors],data_df[target].values)
load_preds = load_CPUmodel.predict(testdf)

print("Fraud:", load_preds)
