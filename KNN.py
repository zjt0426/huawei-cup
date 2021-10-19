import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
train_data = data.iloc[:, 1:730]
X = train_data.values



data_test = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx',sheet_name='test')
test_data = data_test.iloc[:, 1:730]
test_data = test_data.values


for i in range(4):
    labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ADMET.xlsx')
    feature = labels.iloc[:, i+1]
    y = feature.values.reshape(-1, 1).squeeze()
    X_train ,X_test ,y_train ,y_test = train_test_split(X, y, random_state=0)
    print("训练样本数据的大小：{}".format(X_train.shape))
    print("训练样本标签的大小：{}".format(y_train.shape))
    print("测试样本数据的大小：{}".format(X_test.shape))
    print("测试样本标签的大小：{}".format(y_test.shape))

    # 构造KNN模型
    # knn = KNeighborsClassifier(n_neighbors=1)
    knn = KNeighborsClassifier(n_neighbors=3)

    # 训练模型
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # 评估模型
    print("模型精度：{:.2f}".format(np.mean(y_pred == y_test)))
    print("模型精度：{:.2f}".format(knn.score(X_test, y_test)))

    prediction = knn.predict(test_data)
    prediction_prob = knn.predict_proba(test_data)      # 概率形式
    print("预测的目标类别是：{}".format(prediction))       # 0 1 形式
    print(prediction)

    out = pd.DataFrame(prediction)
    out.to_excel(r'D:\dessktop\数学建模\ClassifierKNN_%s.xlsx' % i)