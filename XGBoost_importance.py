from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBRegressor,XGBClassifier
import matplotlib.pylab as plt
import seaborn as sns
import lightgbm as lgb
import pandas as pd
import numpy as np

data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
feature_train = data.iloc[:, 1:730]
# X = train_data.values


labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
label = labels.iloc[:, 2]
# labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ADMET.xlsx')
# label = labels.iloc[:, 1]



def plot_importance(model,X,y):
    model.fit(X,y)
    importance=model.feature_importances_
    if importance.ndim==2:
        importance=importance[0]
    # for i,v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f'%(i,v))
    plt.bar([*range(len(importance))],importance)
    plt.savefig('D:\dessktop\数学建模\XGBsortimportance1.png')
    plt.show()

    indices = np.argsort(importance)[::-1]
    feat_labels = data.columns
    for f in range(feature_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))

# model = DecisionTreeRegressor()
# plot_importance(model,X_reg,y_reg) # 选出了3个重要特征
#
# model = DecisionTreeClassifier()
# plot_importance(model,X_class,y_class) #选出了4个重要特征#
#
# model = RandomForestRegressor()
# plot_importance(model,X_reg,y_reg) # 选出了2-3个重要特征
#
# model = RandomForestClassifier()
# plot_importance(model,X_class,y_class) #选出了2-3个重要特征#


model = XGBRegressor()
plot_importance(model,feature_train,label)

# model = XGBClassifier()
# plot_importance(model,X_class,y_class) #选出了7个重要特征