from sklearn.externals import joblib
import  pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from  constants import  *
if __name__ == '__main__':
    test_data = pd.read_csv(test_offline_data_path)

    offline_user_feature = pd.read_csv('Resource/features/offline/trainUserFeature/user_feature.csv')
    offline_mer_feature = pd.read_csv('Resource/features/offline/trainMerFeature/merchant_feature.csv')
    offline_user_mer_feature = pd.read_csv('Resource/features/offline/trainUserMerFeature/user_merchant_feature.csv')

    online_user_feature = pd.read_csv('Resource/features/online/trainUserFeature/user_feature.csv')
    # online_mer_feature = pd.read_csv('Resource/features/online/trainMerFeature/merchant_feature.csv')
    # online_user_mer_feature = pd.read_csv('Resource/features/online/trainUserMerFeature/user_merchant_feature.csv')

    user_feature = offline_user_feature.merge(online_user_feature,on='userId',how='left')

    columns  = test_data.columns.tolist()
    df = test_data.merge(user_feature,on='userId',how = 'left')
    df1 =df.merge(offline_user_mer_feature,on=['userId','merchantId'],how = 'left')

    df2 = df1.merge(offline_mer_feature,on='merchantId',how = 'left')
    df2.fillna(np.nan, inplace=True)
    df2.drop(columns,axis=1,inplace = True)
    df2.to_csv('Resource/train_features.csv',index = False)

    train_features = pd.read_csv(train_feature_path).astype(float)
    columns = train_features.columns.tolist()
    for col in columns:
        if col is 'userAverageDistance_y' or col is 'userAverageDistance_x':
            train_features[col] = train_features[col].fillna(-1)
        else:
            train_features[col] = train_features[col].fillna(0)

    # train_features.to_csv(train_feature_filna_path)

    max_abs_scaler = MaxAbsScaler()
    x_test_maxabs = max_abs_scaler.fit_transform(train_features)
    clf = joblib.load(model_path+"/linear_0.1")
    y_pre = clf.predict(x_test_maxabs)

    y_pre_df = pd.DataFrame(pd.Series(y_pre), columns=['Probability'])

    print(test_data)
    submit_df = test_data[['userId','Coupon_id','Date_received']].join(y_pre_df)

    submit_df.rename(columns={'userId':'User_id'})
    submit_df.to_csv('submit/submit.csv',index=False)
    print(y_pre)
