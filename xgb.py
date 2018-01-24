import xgboost
import pandas as pd
import numpy as np
from  constants import  *
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from  datetime import  datetime
from sklearn.externals import joblib
def cal_average_auc(df):
    grouped = df.groupby('Coupon_id', as_index=False).apply(lambda x: calc_auc(x))
    return grouped['auc'].mean(skipna=True)


def calc_auc(df):
    coupon = df['Coupon_id'].iloc[0]
    y_true = df['label'].values
    if len(np.unique(y_true)) != 2:
        auc = np.nan
    else:
        y_pred = df['predict'].values
        auc = roc_auc_score(np.array(y_true), np.array(y_pred))
    return pd.DataFrame({'Coupon_id': [coupon], 'auc': [auc]})

if __name__ == '__main__':
    start_time = datetime.now()
    train_features = pd.read_csv(less_train_feature).astype(float)
    train_label = pd.read_csv(less_train_label)

    x_train,x_test,y_train,y_test = train_test_split(train_features,train_label,
                                                     random_state=1,train_size=0.8)
    max_abs_scaler = MaxAbsScaler()
    x_train_maxabs = max_abs_scaler.fit_transform(x_train.values)
    x_test_maxabs = max_abs_scaler.fit_transform(x_test.values)
    y_train_label = y_train['label'].astype(float)
    y_test_label = y_test['label'].astype(float)

    # test_data = pd.read_csv(test_offline_data_path)
    # offline_user_feature = pd.read_csv('Resource/features/offline/trainUserFeature/user_feature.csv')
    # offline_mer_feature = pd.read_csv('Resource/features/offline/trainMerFeature/merchant_feature.csv')
    # offline_user_mer_feature = pd.read_csv('Resource/features/offline/trainUserMerFeature/user_merchant_feature.csv')
    # online_user_feature = pd.read_csv('Resource/features/online/trainUserFeature/user_feature.csv')
    # user_feature = offline_user_feature.merge(online_user_feature, on='userId', how='left')
    # columns = test_data.columns.tolist()
    # df = test_data.merge(user_feature, on='userId', how='left')
    # df1 = df.merge(offline_user_mer_feature, on=['userId', 'merchantId'], how='left')
    # df2 = df1.merge(offline_mer_feature, on='merchantId', how='left')
    # df2.fillna(np.nan, inplace=True)
    # df2.drop(columns, axis=1, inplace=True)
    # df2.to_csv('resource/train_features.csv', index=False)

    print(x_train)
    print('--------------------------------------')
    train_matrix = xgboost.DMatrix(x_train.values,label = y_train_label.values,feature_names=x_train.columns)
    test_matrix = xgboost.DMatrix(x_test.values,label = y_test_label.values,feature_names=x_test.columns)

    # print(train_matrix)
    # predict_matrix = xgboost.DMatrix(df2.values,feature_names=df2.columns)

    watch_list = [(train_matrix,'train'),(test_matrix,'eval')]
    params = {
        'max_depth': 8,
        'eta': 0.1,
        'silent': 1,
        'seed': 13,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 100,
        'max_delta_step': 20
    }
    model = xgboost.train(params,train_matrix,num_boost_round=1000,evals=watch_list,early_stopping_rounds=50)

    train_pred_labels = model.predict(train_matrix,ntree_limit=model.best_ntree_limit)

    test_pred_labels = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)
    y_train_df = pd.DataFrame(pd.Series(train_pred_labels), columns=['predict'])
    y_test_df = pd.DataFrame(pd.Series(test_pred_labels), columns=['predict'])

    y_train_reindex = y_train.reset_index(drop=True)
    y_test_reindex = y_test.reset_index(drop=True)

    train_evaluete = y_train_reindex[['Coupon_id', 'label']].join(y_train_df)
    test_evalute = y_test_reindex[['Coupon_id', 'label']].join(y_test_df)
    print('测试集平均auc: ', cal_average_auc(test_evalute))
    print('训练集平均auc: ', cal_average_auc(train_evaluete))

    end_time = datetime.now()
    diff_time = end_time - start_time
    joblib.dump(model, 'model/xgb/xgb_model.pkl')
    print('运行时间 ：', diff_time.seconds / 60)

