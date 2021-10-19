import numpy as np
import pandas as pd
from pandas import set_option
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


data = pd.read_excel(r'D:\dessktop\GT\final.xlsx')
print(data.columns)

# data = pd.read_csv(r'D:\dessktop\数学建模\ss.csv')
train_data_ = data.iloc[:, :]
print(train_data_.head())
train_data = train_data_.values



set_option('display.width', 120)
# print(data.head(30))
set_option('precision', 2)  # 小数
# print(data.corr(method='pearson'))
# data.hist()    # 直方图
# plt.show()
# data.plot(kind='density', subplots=True, layout=(4, 5), sharex=False, fontsize=1)    # 密度图
# plt.savefig(r'D:\dessktop\GT\result.png')
# plt.show()
# data.plot(kind='box', subplots=True, layout=(4, 5), sharex=False, sharey=False, fontsize=8)   # 箱型图
# plt.savefig(r'D:\dessktop\GT\result.png')
# plt.show()
# scatter_matrix(data)
# plt.savefig(r'D:\dessktop\GT\result.png')
# plt.show()
# plt.figure(figsize=(10, 8))
# new_df = data.corr()
# sns.heatmap(new_df, annot=True, vmax=1, square=True)
# plt.savefig(r'D:\sim\heatmap.png')
# plt.show()