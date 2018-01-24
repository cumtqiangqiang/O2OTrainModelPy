import xgboost
import pandas as pd
import numpy as np
from  constants import  *
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from  datetime import  datetime
from sklearn.externals import joblib


if __name__ == '__main__':
    test_data = pd.read_csv(test_offline_data_path)

    offline_user_feature = pd.read_csv('Resource/features/offline/trainUserFeature/user_feature.csv')
    offline_mer_feature = pd.read_csv('Resource/features/offline/trainMerFeature/merchant_feature.csv')
    offline_user_mer_feature = pd.read_csv('Resource/features/offline/trainUserMerFeature/user_merchant_feature.csv')

    online_user_feature = pd.read_csv('Resource/features/online/trainUserFeature/user_feature.csv')
    # online_mer_feature = pd.read_csv('Resource/features/online/trainMerFeature/merchant_feature.csv')
    # online_user_mer_feature = pd.read_csv('Resource/features/online/trainUserMerFeature/user_merchant_feature.csv')

    user_feature = offline_user_feature.merge(online_user_feature, on='userId', how='left')
    columns = test_data.columns.tolist()
    df = test_data.merge(user_feature, on='userId', how='left')
    df1 = df.merge(offline_user_mer_feature, on=['userId', 'merchantId'], how='left')
    df2 = df1.merge(offline_mer_feature, on='merchantId', how='left')
    df2.fillna(np.nan, inplace=True)
    df2.drop(columns, axis=1, inplace=True)
    df2.to_csv('resource/train_features.csv', index=False)

    # train_features = pd.read_csv(train_feature_path).astype(float)
    columns = df2.columns.tolist()
    for col in columns:
        if col is 'userAverageDistance_y' or col is 'userAverageDistance_x':
            df2[col] = df2[col].fillna(-1)
        else:
            df2[col] = df2[col].fillna(0)

    max_abs_scaler = MaxAbsScaler()
    x_test_maxabs = max_abs_scaler.fit_transform(df2)
    clf = joblib.load('model/xgb/xgb_model.pkl')
    test_matrix = xgboost.DMatrix(df2.values,feature_names=df2.columns)

    y_pre = clf.predict(test_matrix)
    y_pre_df = pd.DataFrame(pd.Series(y_pre), columns=['Probability'])
    submit_df = test_data[['userId', 'Coupon_id', 'Date_received']].join(y_pre_df)
    submit_df.rename(columns={'userId': 'User_id'})
    submit_df.to_csv('submit/submit.csv', index=False)