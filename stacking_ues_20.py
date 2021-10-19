from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

import pandas as pd

# x,y = make_classification(n_samples=6000)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
train_data = data.iloc[:, 1:730]
X = train_data.values

data_test = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx',sheet_name='test')
test_data = data_test.iloc[:, 1:730]
test_data = test_data.values

for m in range(5):
    labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ADMET.xlsx')
    feature = labels.iloc[:, m+1]
    # y = feature.values.reshape(-1, 1).squeeze()
    y = feature.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    ### 第一层模型
    clfs = [GBDT(n_estimators=100),
           RF(n_estimators=100),
           ET(n_estimators=100),
           ADA(n_estimators=100)
    ]
    X_train_stack  = np.zeros((X_train.shape[0], len(clfs)))
    X_test_stack = np.zeros((X_test.shape[0], len(clfs)))

    ### 6折stacking
    n_folds = 6
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for i,clf in enumerate(clfs):
    #print("分类器：{}".format(clf))
        X_stack_test_n = np.zeros((X_test.shape[0], n_folds))
        for j,(train_index,test_index) in enumerate(skf.split(X_train,y_train)):
                    tr_x = X_train[train_index]
                    tr_y = y_train[train_index]
                    clf.fit(tr_x, tr_y)
                    #生成stacking训练数据集
                    X_train_stack [test_index, i] = clf.predict_proba(X_train[test_index])[:,1]
                    X_stack_test_n[:,j] = clf.predict_proba(X_test)[:,1]
        #生成stacking测试数据集
        X_test_stack[:,i] = X_stack_test_n.mean(axis=1)


    ###第二层模型LR
    clf_second = LogisticRegression(solver="lbfgs")
    clf_second.fit(X_train_stack,y_train)
    pred = clf_second.predict_proba(X_test_stack)[:,1]
    # print(X_test_stack)
    # print(pred)
    # print(pred.shape)
    print(roc_auc_score(y_test,pred))#0.9946


    # output = clf_second.predict(test_data)
    # print(output)
    # prediction_prob = cnn_knn.predict_proba(test_data)      # 概率形式
    # print("预测的目标类别是：{}".format(output))
    #
    # out = pd.DataFrame(output)
    # out.to_excel(r'D:\dessktop\数学建模\Stacking_all_%s.xlsx' % m)



    ###GBDT分类器
    clf_1 = clfs[0]
    clf_1.fit(X_train,y_train)
    pred_1 = clf_1.predict_proba(X_test)[:,1]
    print(roc_auc_score(y_test,pred_1))#0.9922

    output = clf_1.predict(test_data)
    # print(output)
    # prediction_prob = cnn_knn.predict_proba(test_data)      # 概率形式
    # print("预测的目标类别是：{}".format(output))

    out = pd.DataFrame(output)
    out.to_excel(r'D:\dessktop\数学建模\Stacking_all_%s.xlsx' % m)

    ###随机森林分类器
    clf_2 = clfs[1]
    clf_2.fit(X_train,y_train)
    pred_2 = clf_2.predict_proba(X_test)[:,1]
    print(roc_auc_score(y_test,pred_2))#0.9944

    ###ExtraTrees分类器
    clf_3 = clfs[2]
    clf_3.fit(X_train,y_train)
    pred_3 = clf_3.predict_proba(X_test)[:,1]
    print(roc_auc_score(y_test,pred_3))#0.9930

    ###AdaBoost分类器
    clf_4 = clfs[3]
    clf_4.fit(X_train,y_train)
    pred_4 = clf_4.predict_proba(X_test)[:,1]
    print(roc_auc_score(y_test,pred_4))#0.9875

    import joblib
    import pickle
    joblib.dump(value=clf_second, filename='D:\dessktop\数学建模\weights\Stacking%s.pkl' % m)

    # 下载本地模型
    # new_model = joblib.load(filename="D:\dessktop\数学建模\weights\Stacking.pkl")
    # predict = new_model.predict(test)


    # In = [-0.020853473, 3.15561696, -0.005032004, 30, 0, 15.9959237, 0 , 0.090909091,
    #       4.860942871, 0, -0.035432878, -0.361398872, 0, 0.284152737, 0.166666667, 10, 0.17241]