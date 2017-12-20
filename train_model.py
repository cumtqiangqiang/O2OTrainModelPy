import numpy as np
from sklearn.model_selection import train_test_split
import  pandas as pd
from  constants import  *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from memory_profiler import profile
import gc

if __name__ == '__main__':

    label_data = pd.read_csv(label_data_path)
    train_features = pd.read_csv(train_feature_path).astype(float)

    y_label = label_data['label'].astype(float)
    # train_features['label'] = pd.Series(y_label.ravel())

    columns = train_features.columns.tolist()
    for col in columns:
        if col is 'userAverageDistance_y' or col is 'userAverageDistance_x':
            train_features[col] = train_features[col].fillna(-1)
        else:
            train_features[col] = train_features[col].fillna(0)

    gc.collect()
    x_train,x_test,y_train,y_test =train_test_split(train_features,y_label,
                                                    random_state=1,train_size=0.8)

    # print(x_train.info())
    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=3)),
                   ('clf', LogisticRegression())])

    lr.fit(x_train, y_train.ravel())

    y_hat = lr.predict(x_test)
    print(print('准确度：%.2f%%' % (100*np.mean(y_hat == y_test.ravel()))))





