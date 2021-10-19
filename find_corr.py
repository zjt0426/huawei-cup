import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_regression



# data = pd.read_excel(r'D:\dessktop\数学建模\lgb_xgb.xlsx')
data = pd.read_excel(r'D:\dessktop\GT\final.xlsx')
train_data = data.iloc[:, :]
# data = data.iloc[1:, :]
print(data.shape)




corr = data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.savefig(r'D:\dessktop\GT\final4.png')
plt.show()

