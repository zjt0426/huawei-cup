import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from scipy.stats import pearsonr

data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
data = data.iloc[:, 1:730]
labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
feature = labels.iloc[:, 1]
feature = feature.values.reshape(-1, 1).squeeze()
# feature = feature.values.reshape(-1, 1)
print(feature)
print(type(feature))
print(feature.shape)

# data = data[data.apply(np.sum, axis=1)!=0]
# print(data.shape)

data_ = data.copy()


transfer = VarianceThreshold(threshold=1)
new_data1 = transfer.fit_transform(data_)
var_index = transfer.get_support(True).tolist()
data1 = data_.iloc[:, var_index]
print(data1.head(0))
print(data1.columns)
print(len(data1.head(0)))
# 1974*224
print(data1.head)
print(data1.shape)
train_data = data1.values.reshape(-1, data1.shape[1])
print(train_data)
print(train_data.shape)



#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
# SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=30).fit_transform(data, feature)
# skb = SelectKBest(score_func=lambda X, Y: tuple(map(tuple,array(list(map(lambda x:pearsonr(x,Y),X.T))).T)), k=224).fit_transform(train_data, feature)
# skb = SelectKBest(f_regression, k=20).fit_transform(data1, feature)
# print(skb)

# best_features = SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:pearsonr(x,Y),X.T))).T)), k=20)
best_features = SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x,Y),X.T))).T[0],k=224)
# best_features = SelectKBest(f_regression, k=224)
# fit = best_features.fit(train_data, feature)
fit = best_features.fit(train_data, feature)
print(fit)
print('################')
print(f_regression(data1['naaCH'].values.reshape(-1, 1), feature))


scores = pd.DataFrame(fit.scores_)
print(scores)
print(scores.shape)
# columns = pd.DataFrame(iris.data.columns)
columns = pd.DataFrame(data1.columns)

df_feature_scores = pd.concat([columns, scores], axis=1)
# 定义列名
df_feature_scores.columns = ['Feature', 'Score']
# 按照score排序
df_feature_scores.sort_values(by='Score', ascending=False)

df_feature_scores.to_excel(r'D:\dessktop\数学建模\2021年D题\pearson20.xlsx')


print(df_feature_scores)
print('--------------------------')

# res = skb[0]
# ans = []
# data2array = list(data1.iloc[0, :])
# for i in res:
#     ans.append(data2array.index(i))
# print(ans)


# dataX = data1.iloc[:, [2, 7, 17, 18, 24, 44, 45, 46, 47, 48, 52, 53, 59, 175, 177, 184, 215, 216, 221, 223]]
# print(dataX.head())
# with open(r'D:\dessktop\数学建模\2021年D题\fregression20.txt', 'w') as t:
#     COL = dataX.columns
#     t.write(str(COL))
# t.close()


# 有点问题
# corr = dataX.corr()
# f,ax = plt.subplots(figsize=(18, 18))
#
# sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# plt.savefig(r'D:\dessktop\数学建模\2021年D题\corr30.jpg')
# plt.show()

dataX.to_excel(r'D:\dessktop\数学建模\2021年D题\train_data.xlsx')
feature.to_excel(r'D:\dessktop\数学建模\2021年D题\labels.xlsx')

train = []
train_data = dataX.values.reshape(-1, 20)
labels = feature


