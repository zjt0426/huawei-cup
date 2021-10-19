import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import csv
from sklearn.feature_selection import VarianceThreshold



# 方法1
data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
dataX = data.iloc[:, 1:730].values
x = dataX.astype('float64')
labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
y = labels.iloc[:, 1].values
print(dataX.shape)
print(pearsonr(dataX[:, 1], y))
data_ = data.iloc[:, 1:730]

transfer = VarianceThreshold(threshold=1)
new_data1 = transfer.fit_transform(data_)
var_index = transfer.get_support(True).tolist()
data1 = data_.iloc[:, var_index]
print(data1.shape)
data1 = data1.values



#############Pearsonr分析#########
for i in range(0,224,1):
    #------------pearsonr（x,y）-------------
    result = pearsonr(data1[:, i], y)
    #------------保存结果--------------------
    with open('D:\dessktop\数学建模\pearson2.csv','a+',encoding='GB18030',newline="")as file_write:
        result_writer = csv.writer(file_write)
        result_writer.writerow(result)

# 方法2
data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')

data = data.iloc[:, 1:730]
feature = data.iloc[:, -1]


from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
# SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=30).fit_transform(data, feature)
skb = SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:pearsonr(x,Y),X.T))).T)), k=30).fit_transform(data, feature)
print(skb)
res = skb[0]
ans = []
data2array = list(data.iloc[0, :])
for i in res:
    ans.append(data2array.index(i))
print(ans)