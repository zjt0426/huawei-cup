from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



data = pd.read_excel(r'D:\dessktop\数学建模\lgb_xgb.xlsx')
print(data.columns)
train_data_ = data.iloc[:, :]
print(train_data_.head())
train_data = train_data_.values

labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
feature = labels.iloc[:, 2]
feature = feature.values.reshape(-1, 1).squeeze()

data_test = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx',sheet_name='test')
test_data = data_test.loc[:, list(train_data_.columns)]
print(test_data.shape)
print(test_data.head())
test_data = test_data.values



x_train,x_test,y_train,y_test = train_test_split(train_data,feature,test_size = 0.3,random_state = 1)


models=[LinearRegression(),KNeighborsRegressor(),SVR(),Ridge(),Lasso(),MLPRegressor(alpha=20),DecisionTreeRegressor(),ExtraTreeRegressor(),XGBRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),BaggingRegressor()]
models_str=['LinearRegression','KNNRegressor','SVR','Ridge','Lasso','MLPRegressor','DecisionTree','ExtraTree','XGBoost','RandomForest','AdaBoost','GradientBoost','Bagging']
score_=[]



for name,model in zip(models_str,models):
    print('开始训练模型：'+name)
    model=model   #建立模型
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    score=model.score(x_test,y_test)
    score_.append(str(score)[:5])
    print(name +' 得分:'+str(score))

models=[XGBRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),BaggingRegressor()]
for i, key in enumerate(models):
    # 训练数据
    key.fit(x_train,y_train)
    # 使用测试数据进行回归预测
    y_predict_ = key.predict(x_test)

    # 训练数据的预测值
    y_train_predict_ = key.predict(x_train)
    score_ = key.score(x_test,y_test)
    print('{}的准确率:R^2={}'.format(key, score_))
    output = key.predict(test_data)
    print(output)
    print(output.shape)
    output = pd.DataFrame(output)
    output.to_excel(r'D:\dessktop\数学建模\集成学习\{}_{}.xlsx'.format(i, score_))


# param_test1 = {'n_estimators':range(20,81,10)}
# gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=300,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
#                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
