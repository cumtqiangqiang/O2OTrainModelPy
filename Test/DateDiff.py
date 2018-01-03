
from  datetime import  datetime
from  datetime import  date
import pandas as pd
import  numpy as np
from constants import  *
if __name__ == '__main__':
    # start = '20150607'
    # end = '20160607'
    # date_format ='%Y%m%d'
    # start_date = datetime.strptime(start,date_format)
    # end_date = datetime.strptime(end,date_format)
    #
    # print( (end_date - start_date).days)

    # data1 = pd.read_csv(r'C:\Users\UC227911\Desktop\Mine\O2O\O2OTrainModelPy\Resource\features\offline\trainUserMerFeature\part-00000')
    # print(data1)
    # data2 = pd.read_csv(r'C:\Users\UC227911\Desktop\Mine\O2O\O2OTrainModelPy\Resource\features\offline\trainUserMerFeature\part-00001')
    # data =  data1.append(data2,ignore_index=True)
    # data.to_csv(r'C:\Users\UC227911\Desktop\Mine\O2O\O2OTrainModelPy\Resource\features\offline\trainUserMerFeature\user_merchant_feature.csv',index = False)

    path = '../Resource/features/online/trainUserMerFeature'
    result = None
    sum = 0
    for i in range(15):
        if i < 10:
           file_name = '/part-0000'+str(i)
        else:
            file_name = '/part-000'+str(i)
        path1 = path + file_name
        data = pd.read_csv(path1)
        print(str(i) + ":" +  str(data.shape))
        sum += data.iloc[:,0].size
        if i == 0:
            result = data
        else:
            result = result.append(data)

    print(sum)
    print(result.shape)

    result.to_csv('../Resource/features/online/trainUserMerFeature/user_merchant_feature.csv',index=False)

    # offline_data = pd.read_csv('../Resource/tb01ccf_offline_stage1_train.csv')
    # online_data = pd.read_csv('../Resource/tb02ccf_online_stage1_train.csv')
    #
    # offline_user_set =  set(offline_data['User_id'].tolist())
    # online_user_set = set(online_data['User_id'].tolist())
    # train_on_off_intersection =  offline_user_set & online_user_set
    #
    # print("offline user cnt :" + str(len(offline_user_set)))
    # print("online user cnt :" + str(len(online_user_set)))
    # print("offline & online user cnt :" + str(len(train_on_off_intersection)))
    #
    # test_offline_data = pd.read_csv('../Resource/tb03ccf_offline_stage1_test_revised.csv')
    # test_offline_user_set = set(test_offline_data['User_id'].tolist())
    # test_on_off_intersection =  test_offline_user_set & online_user_set
    #
    # print("test data offline  :" + str(len(test_offline_user_set)))
    # print("test data offline & online user cnt :" + str(len(test_on_off_intersection)))

