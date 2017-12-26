import numpy as np
from sklearn.model_selection import train_test_split
import  pandas as pd
from  constants import  *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import  roc_auc_score
# from memory_profiler import profile
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

    label_data = pd.read_csv(less_label_data)
    train_features = pd.read_csv(less_feature_data).astype(float)


    # train_features['label'] = pd.Series(y_label.ravel())


    x_train,x_test,y_train,y_test =train_test_split(train_features,label_data,
                                                    random_state=1,train_size=0.8)

    y_train_label = y_train['label'].astype(float)
    y_test_label = y_test['label'].astype(float)
    # print(y_label)
    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=3)),
                   ('clf', LogisticRegression())])


    lr.fit(x_train, y_train_label.ravel())

    y_hat = lr.predict(x_test)
    y_pre_df = pd.DataFrame(pd.Series(y_hat), columns=['predict'])

    y_train_reindex = y_train.reset_index(drop=True)
    y_test_reindex = y_test.reset_index(drop=True)


    test_evalute = y_test_reindex[['Coupon_id', 'label']].join(y_pre_df)
    print(test_evalute[['Coupon_id','label','predict']])


    print(cal_average_auc(test_evalute))
    print('准确度：%.2f%%' % (100*np.mean(y_hat == y_test_label.ravel())))





