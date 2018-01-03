import  pandas as pd
from constants import  *
from  datetime import  datetime
import numpy as np
from sklearn.model_selection import train_test_split

def add_label(data):
    received_date = data["Date_received"]
    consume_date = data["Date"]

    label_frame = pd.Series(list(map(lambda x,y: 1. if get_time_diff(x,y) else 0.,
                               received_date,consume_date )))
    label_frame.name = 'label'

    data = data.join(label_frame)

    return data

def get_time_diff(start,end):

    if start != 'null' and end != 'null':
        date_format = '%Y%m%d'
        start_date = datetime.strptime(start, date_format)
        end_date = datetime.strptime(end, date_format)
        if (end_date - start_date).days <= 15 :
            return  True
    else:
        return  False

if __name__ == '__main__':
    # raw_offline_data =  pd.read_csv(train_offline_data_path)
    # data = add_label(raw_offline_data)
    #
    # data.to_csv('Resource/trainLabelData/train_label_data.csv',index=False)
    # label_data = pd.read_csv('Resource/trainLabelData/train_label_data.csv')
    # offline_user_feature = pd.read_csv('Resource/features/offline/trainUserFeature/user_feature.csv')
    # offline_mer_feature = pd.read_csv('Resource/features/offline/trainMerFeature/merchant_feature.csv')
    # offline_user_mer_feature = pd.read_csv('Resource/features/offline/trainUserMerFeature/user_merchant_feature.csv')
    #
    # online_user_feature = pd.read_csv('Resource/features/online/trainUserFeature/user_feature.csv')
    # online_mer_feature = pd.read_csv('Resource/features/online/trainMerFeature/merchant_feature.csv')
    # online_user_mer_feature = pd.read_csv('Resource/features/online/trainUserMerFeature/user_merchant_feature.csv')
    #
    # user_feature = offline_user_feature.merge(online_user_feature,on='userId',how='left')
    #
    # columns  = label_data.columns.tolist()
    # df = label_data.merge(user_feature,on='userId',how = 'left')
    # df1 =df.merge(offline_user_mer_feature,on=['userId','merchantId'],how = 'left')
    #
    # df2 = df1.merge(offline_mer_feature,on='merchantId',how = 'left')
    # df2.fillna(np.nan, inplace=True)
    # df2.drop(columns,axis=1,inplace = True)
    # df2.to_csv('Resource/train_features.csv',index = False)

    train_features = pd.read_csv(train_feature_path).astype(float)
    columns = train_features.columns.tolist()
    for col in columns:
        if col is 'userAverageDistance_y' or col is 'userAverageDistance_x':
            train_features[col] = train_features[col].fillna(-1)
        else:
            train_features[col] = train_features[col].fillna(0)

    train_features.to_csv(train_feature_filna_path,index = False)

















