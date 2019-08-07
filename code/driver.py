#!/usr/bin/python3

import concurrent.futures
import requests
import json
import time
from datetime import datetime
import time
import joblib

def send_post(thread_count,loop_count):
    myvalid = '{ "use_scoring" : true, "scoring_args" : {"transaction": "0,-1.359807134,-0.072781173,2.536346738,1.378155224,-0.33832077,0.462387778,0.239598554,0.098697901,0.36378697,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,0.207971242,0.02579058,0.40399296,0.251412098,-0.018306778,0.277837576,-0.11047391,0.066928075,0.128539358,-0.189114844,0.133558377,-0.021053053,149.62,0"} }'
    myfraud = '{ "use_scoring" : true, "scoring_args" : { "transaction": "406,-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919,0,0"} }'

    url = "http://172.30.225.52:10021/xgboostfraud/1/predict"
    headers = {
        'Content-Type': "application/json,application/json",
        'X-AUTH-TOKEN': "9nVYzAk0wf",
        'Host': "172.30.225.52:10021",
        'cache-control': "no-cache"
        }
    count = 0
    fraudflag = 0
    while (count <= loop_count):
        if fraudflag == 0:
            mydata = myfraud
            fraudflag = 1
        else:
            mydata = myvalid
            fraudflag = 0

        PostStartTime = datetime.now()
        response = requests.request("POST", url, data=mydata, headers=headers)
        PostEndTime = datetime.now()
        count = count + 1
# really nead to parse result for result
        print ("Thread: "+ str(thread_count) + " Transaction: " + str(count) + " Exec Time: " + str(PostEndTime - PostStartTime) + "\n" + response.text)
    return_text = "Thread: " + str(thread_count) + " Elapsed Time: " + str(PostEndTime - PostStartTime)
    return return_text

JobStartTime = datetime.now()

# once working command line or json file input
# max_worker(concurrency), url, X-AUTH-TOKEN, Host, max_duration, max_transactions, sleeptime

max_worker = 3
thread_count = 3
trans_count = 10
ex =  concurrent.futures.ThreadPoolExecutor(max_workers=max_worker)

wait_for_auth = [
    ex.submit(send_post,i,trans_count)
    for i in range(0,thread_count,1)
]

for f in concurrent.futures.as_completed(wait_for_auth):
    print('Result: {}'.format(f.result()))

JobEndTime = datetime.now()
#need transactions submitted, average time, good, bad counts........
print ("\nWith concurrency of: " + str(max_worker) + " Elapsed Time: " + str( JobEndTime - JobStartTime))

