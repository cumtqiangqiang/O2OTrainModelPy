import  numpy as np
import  pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from  constants import  *
if __name__ == '__main__':

    label_data = pd.read_csv(less_label_data)
    data = pd.read_csv(less_feature_data).astype(float)

    x_train, x_test, y_train, y_test = train_test_split(data, label_data['label'].astype(float), random_state=1, train_size=0.6)

    # 分类器
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    # 准确率
    print(clf.score(x_train, y_train))  # 精度
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))