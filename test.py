from scipy.stats import pearsonr
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns

data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
# data = data.T
# all training data
dataX = data.iloc[:, 1:]
print(dataX.head())

data1 = dataX.copy()
# data_ = dataX.copy()
# 过滤掉方差小于0.1的特征
# transfer = VarianceThreshold(threshold=0.1)
# new_data1 = transfer.fit_transform(data_)
# get_support得到的需要留下的下标索引
# var_index = transfer.get_support(True).tolist()
# data1 = data_.iloc[:,var_index]
# print(data1.head())

pear_num = [] #存系数
pear_name = [] #存特征名称
feature_names = data1.columns.tolist()
print(len(feature_names))
#得到每个特征与SalePrice间的相关系数
for i in range(0,len(feature_names)-1):
    print('%s和%s之间的皮尔逊相关系数为%f'%(feature_names[i],feature_names[-1],pearsonr(data1[feature_names[i]],data1[feature_names[-1]])[0]))
    if (abs(pearsonr(data1[feature_names[i]],data1[feature_names[-1]])[0])>0.5):
        pear_num.append(pearsonr(data1[feature_names[i]],data1[feature_names[-1]])[0])
        pear_name.append(feature_names[i])
print(pear_name)
print(len(pear_name))


print(dataX.corr()[u'IC50_nM']) #只显示“时间序列”与其他传感器数据的相关系数
corr = dataX.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.savefig(r'D:\dessktop\数学建模\2021年D题\corr.jpg')
plt.show()

