import  pandas as pd
import  numpy as np
from  datetime import  datetime
import  time
import  re
from constants import *

if __name__ == '__main__':
    train_features = pd.read_csv('./train/train_features.csv').astype(float)
    # train_label = pd.read_csv(less_train_label)
    print(train_features.values)


    # start_time = datetime.now()
    # end_time = datetime.now()
    #
    # diff_time = end_time - start_time
    #
    # print(diff_time.seconds/60)
    # a1 = 'trxIdabcId'
    # p1 = r'[A-Z]'
    # match = re.compile(p1)
    # arr =  match.findall(a1)
    # set = list(set(arr))
    # # print(arr)
    #
    # # result = match.split(a1)
    # # print(result)
    # result = ''
    # for a in  set:
    #     lower_a = a.lower()
    #     replace = '_'+lower_a
    #
    #     result,n  = match.subn(replace,a1)
    #
    # print(result)



