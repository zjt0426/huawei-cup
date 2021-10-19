import numpy as np
import pandas as pd
from pandas import set_option
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor


data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
data = data.iloc[:, 1:730]
feature = data.iloc[:, -1]


data_ = data.copy()
transfer = VarianceThreshold(threshold=0.5)
new_data1 = transfer.fit_transform(data_)
var_index = transfer.get_support(True).tolist()
data1 = data_.iloc[:, var_index]
print(data1.shape)
print(data1.head())



# data1.hist()    # 直方图
# plt.show()
# data1.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, fontsize=1)    # 密度图
# plt.show()
# data1.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, fontsize=8)   # 箱型图
# plt.show()
# scatter_matrix(data1)
# plt.show()
# plt.figure(figsize=(10, 8))

# print(data.iloc[:, ])
names = data1.columns
print(names)


rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
# 单独采用每个特征进行建模，并进行交叉验证
for i in range(261):
    with open(r'D:\dessktop\数学建模\2021年D题\scores.txt', 'w') as f:

        score = cross_val_score(rf, data1.iloc[:, i:i + 1], feature, scoring="r2",  # 注意X[:, i]和X[:, i:i+1]的区别
                                cv=ShuffleSplit(len(data1), 3, .3))
        scores.append((format(np.mean(score), '.3f'), names[i]))
        f.write(str(scores))
print(sorted(scores, reverse=True))
f.close()

