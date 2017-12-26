import  numpy as np
import  pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from  constants import  *
from sklearn.metrics import  roc_auc_score
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
    data = pd.read_csv(less_feature_data).astype(float)


    x_train, x_test, y_train, y_test = train_test_split(data, label_data,
                                                        random_state=1, train_size=0.8)

    y_train_label = y_train['label'].astype(float)
    y_test_label = y_test['label'].astype(float)

    # 分类器
    clf_param = (('linear', 0.1), ('linear', 0.5), ('linear', 1), ('linear', 2),
                 ('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                 ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100))
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    for i, param in enumerate(clf_param):
        clf = svm.SVC(C=param[1], kernel=param[0])
        if param[0] == 'rbf':
            clf.gamma = param[2]
            print('高斯核，C=%.1f，$\gamma$ =%.1f' % (param[1], param[2]))
        else:
            print('线性核，C=%.1f' % param[1])
        clf.fit(x_train, y_train_label.ravel())

        y_pre = clf.predict(x_test)
        y_pre_df = pd.DataFrame(pd.Series(y_pre),columns=['predict'])

        y_train_pre = clf.predict(x_train)
        y_train_df = pd.DataFrame(pd.Series(y_train_pre),columns=['predict'])

        y_train_reindex = y_train.reset_index(drop=True)
        y_test_reindex = y_test.reset_index(drop=True)

        test_evalute = y_test_reindex[['Coupon_id','label']].join(y_pre_df)

        train_evaluete = y_train_reindex[['Coupon_id','label']].join(y_train_df)

        print(cal_average_auc(test_evalute))
        print(cal_average_auc(train_evaluete))

        # print(y_pre)

        # 准确率
        print(clf.score(x_train, y_train_label))  # 精度
        print('训练集准确率：', accuracy_score(y_train_label, clf.predict(x_train)))
        print(clf.score(x_test, y_test_label))
        print('测试集准确率：', accuracy_score(y_test_label, clf.predict(x_test)))
        print('--------------------------------------------------------------')
