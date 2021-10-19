import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc



data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
feature_train = data.iloc[:, 1:730]
# X = train_data.values


labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
label = labels.iloc[:, 2]





params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.02,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',
          "random_state": 2019,
          }
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(feature_train, label, test_size=0.2, random_state=3)

# 创建成lgb特征的数据集格式,将使加载更快
lgb_train = lgb.Dataset(X_train, label=Y_train)
lgb_eval = lgb.Dataset(x_test, label=y_test, reference=lgb_train)

clf = lgb.train(params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                num_boost_round=50,
                verbose_eval=200,
                early_stopping_rounds=200)

importance = clf.feature_importance()
if importance.ndim == 2:
    importance = importance[0]
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f'%(i,v))
plt.bar([*range(len(importance))], importance)
plt.savefig('D:\dessktop\数学建模\LGBsortimportance.png')
plt.show()



indices = np.argsort(importance)[::-1]

feat_labels = data.columns
col1 = []
col2 = []
for f in range(feature_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))
    if f <= 60:
        col1.append(feat_labels[indices[f]])
        col2.append(importance[indices[f]])
my_d = {'column':col1, 'importance':col2}
my_data = pd.DataFrame(my_d)
print(col1)
print(col2)
print(my_data)

find_feature = col1[0:30]
savedata = data.loc[:, find_feature]


savedata.to_excel(r'D:\dessktop\GT\final.xlsx')
print(feature_train.columns)

# feature_names_pd = pd.DataFrame({'column': feature_train.columns,
#                                      'importance': clf.feature_importance(),
#                                      }).sort_values(by='importance', ascending=False)
#
# print(feature_names_pd.sort_values(by='importance'))

# cols = my_data[["column", "importance"]].groupby("column").mean().sort_values(by="importance",
#                                                                                        ascending=False)[:50].index
#
# best_features = my_data.loc[my_data.column.isin(cols)]

plt.figure(figsize=(10, 15))
# sns.barplot(x="importance", y="column", data=best_features.sort_values(by="importance", ascending=False))
sns.barplot(x="importance", y="column", data=my_data)
plt.title('LightGBM Features')
plt.tight_layout()
plt.savefig('D:\dessktop\数学建模\llllllllllll.png')
plt.show()



'''
indices = np.argsort(importance)[::-1]
feat_labels = data.columns
for f in range(feature_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False).index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
plt.figure(figsize=(8, 10))
sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('D:\dessktop\数学建模\lgb_importances1111.png')
'''

'''
importance = clf.feature_importance()
if importance.ndim == 2:
    importance = importance[0]
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f'%(i,v))
plt.bar([*range(len(importance))], importance)
plt.savefig('D:\dessktop\数学建模\LGBsortimportance.png')
plt.show()

indices = np.argsort(importance)[::-1]
feat_labels = data.columns
for f in range(feature_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))
'''
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
model = LogisticRegression()
# fit the model
model.fit(feature_train, label)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
plt.savefig('D:\dessktop\数学建模\Logisticimportance.png')
pyplot.show()