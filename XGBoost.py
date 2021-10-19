import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

data = pd.read_excel(r'D:\dessktop\GT\final.xlsx')
# data = pd.read_csv(r'D:\dessktop\数学建模\ss.csv')
train_data = data.iloc[:, :]
X = train_data.values
print(train_data)
print(train_data.shape)
print(train_data.columns)

labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
feature = labels.iloc[:, 2]
y = feature.values
print(feature)




print(X.shape)

fig, ax = plt.subplots(figsize=(10, 5))


model_reg = LinearRegression()
model_reg.fit(X, y)

accuracies = cross_val_score(estimator=model_reg, X=X, y=y, cv=10)
print(accuracies.mean())



def print_cv_params(selecter_param, selecter_param_str, parameters):
    grid_search = GridSearchCV(estimator=model_xgb,
                               param_grid=parameters,
                               scoring='neg_mean_squared_error',
                               cv=10,
                               n_jobs=-1)

    grid_result = grid_search.fit(X, y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    pyplot.errorbar(selecter_param, means, yerr=stds)
    pyplot.title("XGBoost " + selecter_param_str + " vs Mean Squared Error")
    pyplot.xlabel(selecter_param_str)
    pyplot.ylabel('Mean Squared Error')
    pyplot.savefig('D:\dessktop\GT\MSE.png')


model_xgb = XGBRegressor()
n_estimators = range(50, 800, 150)
parameters = dict(n_estimators=n_estimators)

print_cv_params(n_estimators, 'n_estimators', parameters)
learning_rate = np.arange(0.0, 0.2, 0.03)
parameters = dict(learning_rate=learning_rate)

print_cv_params(learning_rate, 'learning_rate', parameters)
max_depth = range(0, 7)
parameters = dict(max_depth=max_depth)

print_cv_params(max_depth, 'max_depth', parameters)
min_child_weight = np.arange(0.5, 2., 0.3)
parameters = dict(min_child_weight=min_child_weight)

print_cv_params(min_child_weight, 'min_child_weight', parameters)
gamma = np.arange(.001, .01, .003)
parameters = dict(gamma=gamma)

print_cv_params(gamma, 'gamma', parameters)
subsample = np.arange(0.3, 1., 0.2)
parameters = dict(subsample=subsample)

print_cv_params(subsample, 'subsample', parameters)
colsample_bytree = np.arange(.6, 1, .1)
parameters = dict(colsample_bytree=colsample_bytree)

print_cv_params(colsample_bytree, 'colsample_bytree', parameters)
parameters = {
    'colsample_bytree': [.6],
    'subsample': [.9, 1],
    'gamma': [.004],
    'min_child_weight': [1.1, 1.3],
    'max_depth': [3, 6],
    'learning_rate': [.15, .2],
    'n_estimators': [1000],
    'reg_alpha': [0.75],
    'reg_lambda': [0.45],
    'seed': [42]
}

grid_search = GridSearchCV(estimator=model_xgb,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=5,
                           n_jobs=-1)

model_xgb = grid_search.fit(X, y)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

accuracies = cross_val_score(estimator=model_xgb, X=X, y=y, cv=10)
accuracies.mean()

data_test = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx',sheet_name='test')

test_data = data_test.loc[:, list(data.columns)]
print(test_data.shape)
print(test_data.head())
test_data = test_data.values

# y_pred = model_xgb.predict(test)
# y_pred = np.floor(np.expm1(y_pred))
# submission = pd.concat([test_ids, pd.Series(y_pred)],
#                        axis=1,
#                        keys=['Id', 'SalePrice'])
# submission.to_csv('sample_submission.csv', index=False)
# submission
#
output = model_xgb.predict(test_data)
print(output)
print(output.shape)
output = pd.DataFrame(output)
output.to_excel(r'D:\dessktop\数学建模\第二问\XGBoost.xlsx')


import joblib
import pickle

joblib.dump(value=model_xgb, filename='D:\dessktop\数学建模\weights\XGBoost.pkl')

# 下载本地模型
# new_model = joblib.load(filename="D:\dessktop\数学建模\weights\XGBoost.pkl")
# predict = new_model.predict(test)
