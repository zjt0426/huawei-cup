import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
import math
from sklearn.metrics import r2_score
import xgboost as xgb



# 用20个特征来训练，同时测试集也取对应20个特征来预测50个药品

# data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\feature_fin.xlsx')
data = pd.read_excel(r'D:\dessktop\数学建模\lgb_xgb.xlsx')
print(data.columns)

# data = pd.read_csv(r'D:\dessktop\数学建模\ss.csv')
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


'''
ensembles = {}
ensembles['ScalerAB'] = Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])
ensembles['ScalerAB-KNN'] = Pipeline([('Scaler', StandardScaler()), ('ABKNN', AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)))])
ensembles['ScalerAB-LR'] = Pipeline([('Scaler', StandardScaler()), ('ABLR', AdaBoostRegressor(LinearRegression()))])
ensembles['ScalerRFR'] = Pipeline([('Scaler', StandardScaler()), ('RFR', RandomForestRegressor())])
ensembles['ScalerETR'] = Pipeline([('Scaler', StandardScaler()), ('ETR', ExtraTreesRegressor())])
ensembles['ScalerRBR'] = Pipeline([('Scaler', StandardScaler()), ('RBR', GradientBoostingRegressor())])

results__ = []

for key in ensembles:
    kfold = KFold(n_splits=10, shuffle=False)
    cv__result = cross_val_score(ensembles[key], train_data, feature, scoring='neg_mean_squared_error', cv=kfold)
    results__.append(cv__result)

    print('%s: %f (%f)' % (key, cv__result.mean(), cv__result.std()))

'''

x_train,x_test,y_train,y_test = train_test_split(train_data, feature,test_size=0.25)

models = {}
models['LR'] = LinearRegression()
models['SVM'] = SVR()
models['Tree'] = DecisionTreeRegressor()
models['Lasso'] = Lasso()
models['En'] = ElasticNet()
models['KNN'] = KNeighborsRegressor()
result = []
for key in models:
    kfold = KFold(n_splits=10, shuffle=False)
    cv_result = cross_val_score(models[key], x_train, y_train, cv=kfold, scoring='r2')
    # neg_mean_squared_error
    result.append(cv_result)
    print(key)
    print(cv_result)
    # print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))
    # print('%s: %f' % (key, cv_result))

tt = DecisionTreeRegressor()
# 训练数据
tt.fit(x_train,y_train)
# 使用测试数据进行回归预测
y_predict_ = tt.predict(x_test)

# 训练数据的预测值
y_train_predict_ = tt.predict(x_train)
score_ = tt.score(x_test,y_test)
print('决策树的准确率:'+r'R^2=%f' % (score_))
output = tt.predict(test_data)
print(output)
print(output.shape)
output = pd.DataFrame(output)
output.to_excel(r'D:\dessktop\数学建模\第二问\RegressionDecisionTree_{}.xlsx'.format(score_))




stand = StandardScaler()#标准化操作
x_train  = stand.fit_transform(x_train)#计算训练数据的均值和方差，基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正态分布
x_test  = stand.fit_transform(x_test)

lr = LinearRegression()
# 训练数据
lr.fit(x_train,y_train)
# 使用测试数据进行回归预测
y_predict = lr.predict(x_test)

# 训练数据的预测值
y_train_predict=lr.predict(x_train)
score = lr.score(x_test,y_test)
print('线性回归模型的准确率:'+r'R^2=%f' % (score))
# print('回归树r2_score:', r2_score(y_test, y_predict))





MSE = mean_squared_error(y_train, y_train_predict)
# 计算RMSE（均方根误差）
RMSE = math.sqrt(MSE)
print('MSE = %.4f'% (MSE))
print('RMSE = %.f' % RMSE)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
train_features, test_features, train_price, test_price = train_test_split(train_data, feature, test_size=0.25)
dtr = DecisionTreeRegressor()
dtr.fit(train_features, train_price)

# 预测测试集中的房价
predict_price = dtr.predict(test_features)
# 测试集的结果评价
print('回归树准确率:', dtr.score(test_features, test_price))
print('回归树r2_score:', r2_score(test_price, predict_price))
print('回归树二乘偏差均值:', mean_squared_error(test_price, predict_price))
print('回归树绝对值偏差均值:', mean_absolute_error(test_price, predict_price))


output = dtr.predict(test_data)
output = pd.DataFrame(output)
output.to_excel(r'D:\dessktop\数学建模\第二问\回归树_{}.xlsx'.format(score_))


from keras import models
from keras import layers


# train_features, test_features, train_price, test_price
data = pd.read_excel(r'D:\dessktop\数学建模\lightgbm.xlsx')
train_data_ = data.iloc[:, 1:]
train_data = train_data_.values
labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
feature = labels.iloc[:, 2]
feature = feature.values
data_test = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx',sheet_name='test')
test_data = data_test.loc[:, list(train_data_.columns)]
test_data = test_data.values

train_features, test_features, train_price, test_price = train_test_split(train_data, feature, test_size=0.25)

stand = StandardScaler()#标准化操作

# train_features = stand.fit_transform(train_features)#计算训练数据的均值和方差，基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正态分布
# test_features = stand.fit_transform(test_features)
# train_price = stand.fit_transform(train_price.reshape(-1, 1))
# test_price = stand.fit_transform(test_price.reshape(-1, 1))


def build_network():
    network = models.Sequential()
    network.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    network.add(layers.BatchNormalization(epsilon=1e-6))
    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.BatchNormalization(epsilon=1e-6))
    network.add(layers.Dense(1))
    network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # 均方误差作为损失
    # network.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 均方误差作为损失
    return network

# k-fold 验证 , 基于numpy实现
import numpy as np

k = 4
num_val_samples = len(train_features)//k
num_epochs = 200
all_scores = []
all_mae_histories = []

for i in range(k):
    print("processing fold #", i)
    # Prepare the validation data
    val_data = train_features[i*num_val_samples: (i+1)*num_val_samples]
    val_targets = train_price[i*num_val_samples: (i+1)*num_val_samples]

    #Prepare the training dataa: data from all other partitions
    partial_train_data = np.concatenate(
                            [train_features[: i*num_val_samples],
                             train_features[(i+1)*num_val_samples:]],
                            axis=0
                            )
    partial_train_targets = np.concatenate(
                            [train_price[: i*num_val_samples],
                             train_price[(i+1)*num_val_samples:]],
                            axis=0
                            )
    network = build_network()
    # 记录每次迭代的训练集和验证集 acc和loss
    history = network.fit(partial_train_data, partial_train_targets, epochs=num_epochs,
                          batch_size=1, verbose=1,
                          validation_data=(val_data, val_targets))
    print(history.history.keys())
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    # val_mse, val_mae = network.evaluate(val_data, val_targets, verbose=1)
    #
    # all_scores.append(val_mae)

# print('all_scores: ', all_scores)
# print('mean ', np.mean(all_scores))
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print('average_mae_history: ', average_mae_history)

# Plotting validation scores
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# -------------------------------
plt.clf()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 根据前面的MAE分析， epochs=80 会是一个合适的模型,下面重新训练模型

model = build_network()
model.fit(train_features, train_price,
          epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_features, test_price)

print('test_mae_score: ', test_mae_score)  # Mean Squared Error

test_features = stand.fit_transform(test_features)

test_data = stand.fit_transform(test_data)
out = model.predict(test_data)
print(model.score(model.predict(test_features), test_price))
print(out)
print(out.shape)
out = pd.DataFrame(out)
out.to_excel(r'D:\dessktop\数学建模\RegressionMLP.xlsx')

model.save(r'D:\dessktop\数学建模\cnn.h5')