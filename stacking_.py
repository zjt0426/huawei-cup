from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LogisticRegression


data = pd.read_excel(r'D:\dessktop\数学建模\lgb_xgb.xlsx')
print(data.columns)
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

x_train,x_test,y_train,y_test = train_test_split(train_data,feature,test_size = 0.3,random_state = 1)

# clf1 = LinearRegression()
clf1 = GradientBoostingRegressor(n_estimators=100)
clf3 = ExtraTreeRegressor()
clf4 = RandomForestRegressor()
clf5 = AdaBoostRegressor()
base_models = [DecisionTreeRegressor, RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, XGBRegressor]
estimators = [clf1, clf3, clf4, clf5]
sclf = StackingRegressor(
    estimators=estimators,
    final_estimator=XGBRegressor()
)


# models=[,KNeighborsRegressor(),,Ridge(),,MLPRegressor(alpha=20)
# ,DecisionTreeRegressor(),,XGBRegressor(),,AdaBoostRegressor(),,BaggingRegressor()]


sclf.fit(x_train,y_train)
y_pred=sclf.predict(x_test)
score=sclf.score(x_test,y_test)
# score_.append(str(score)[:5])
print(' 得分:'+str(score))