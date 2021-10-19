import matplotlib.pylab as plt
import seaborn as sns
import lightgbm as lgb
import pandas as pd
import numpy as np

# data_df = pd.read_csv('train.csv')
# label = data_df['TARGET']
# feature = data_df.drop(['TARGET', 'ID'], axis=1)
# print(feature.shape)
# data_test = pd.read_csv('test.csv')
# data_test_ID = data_test['ID']
# data_test_feature = data_test.drop(['ID'], axis=1)
# print(data_test_feature.shape)
# feature_all = pd.concat([feature, data_test_feature])
#
# feature_all = pd.get_dummies(feature_all,
#                              dummy_na=True,
#                              columns=None)
#
# feature_train = feature_all.iloc[:len(feature), :]
# print(feature_train.shape)
# feature_test = feature_all.iloc[len(feature):]
# print(feature_test.shape)


data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
feature_train = data.iloc[:, 1:730]
# X = train_data.values

# labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ADMET.xlsx')
# label = labels.iloc[:, 1]
# y = feature.values.reshape(-1, 1).squeeze()

labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ERα_activity.xlsx')
label = labels.iloc[:, 2]


# feature = feature.drop(['poutcome_nan'],axis=1)  # 根据特征重要性进行选择，保留几乎全部特征得分最高


# 训练模型
def train_model(data_X, data_y):
    from sklearn.model_selection import train_test_split
    X_train, x_test, Y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=3)

    # 创建成lgb特征的数据集格式,将使加载更快
    lgb_train = lgb.Dataset(X_train, label=Y_train)
    lgb_eval = lgb.Dataset(x_test, label=y_test, reference=lgb_train)
    # binary
    parameters = {
        'task': 'train',
        'max_depth': 15,
        'boosting_type': 'gbdt',
        'num_leaves': 20,  # 叶子节点数
        'n_estimators': 50,
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.2,
        'feature_fraction': 0.7,  # 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
        'bagging_fraction': 1,  # 类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
        'bagging_freq': 3,  # bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
        'lambda_l1': 0.5,
        'lambda_l2': 0,
        'cat_smooth': 10,  # 用于分类特征,这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
        'is_unbalance': False,  # 适合二分类。这里如果设置为True，评估结果降低3个点
        'verbose': 0
    }

    evals_result = {}  # 记录训练结果所用
    gbm_model = lgb.train(parameters,
                          lgb_train,
                          valid_sets=[lgb_train, lgb_eval],
                          num_boost_round=50,  # 提升迭代的次数
                          early_stopping_rounds=5,
                          evals_result=evals_result,
                          verbose_eval=10
                          )

    prediction = gbm_model.predict(x_test, num_iteration=gbm_model.best_iteration)
    from sklearn.metrics import roc_auc_score
    roc_auc_score = roc_auc_score(y_test, prediction)
    print(roc_auc_score)
    return gbm_model, evals_result


model, evals_result = train_model(feature_train, label)

# 运行结果


# 可视化训练结果以及模型下的特征重要性
def lgb_importance():
    model, evals_result = train_model(feature_train, label)

    ax = lgb.plot_metric(evals_result, metric='auc')  # metric的值与之前的params里面的值对应
    plt.title('metric')
    plt.show()

    feature_names_pd = pd.DataFrame({'column': feature_train.columns,
                                     'importance': model.feature_importance(),
                                     }).sort_values(by='importance')

    cols = feature_names_pd[["column", "importance"]].groupby("column").mean().sort_values(by="importance",
                                                                                           ascending=False)[:50].index

    best_features = feature_names_pd.loc[feature_names_pd.column.isin(cols)]

    plt.figure(figsize=(10, 15))
    sns.barplot(x="importance", y="column", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('D:\dessktop\数学建模\lightgbm_importance.png')
    plt.show()

lgb_importance()



