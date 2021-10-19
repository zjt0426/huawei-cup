import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, LeakyReLU, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from keras.initializers import glorot_normal



data = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\Molecular_Descriptor.xlsx')
train_data = data.iloc[:, 1:730]
X = train_data.values.tolist()
X = np.expand_dims(X, axis=0)

labels = pd.read_excel(r'D:\dessktop\数学建模\2021年D题\ADMET.xlsx')
feature = labels.iloc[:, 1]
y = feature.values.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Conv1D(8, 11, kernel_initializer=glorot_normal(seed=1), input_shape=(729, 1)))
model.add(LeakyReLU(alpha=0.05))
model.add(Conv1D(16, 9, kernel_initializer=glorot_normal(seed=1)))
model.add(LeakyReLU(alpha=0.05))
model.add(Conv1D(32, 5, kernel_initializer=glorot_normal(seed=1)))
model.add(LeakyReLU(alpha=0.05))
model.add(Conv1D(64, 3, kernel_initializer=glorot_normal(seed=1)))
model.add(LeakyReLU(alpha=0.05))
model.add(Flatten())
model.add(Dense(64, kernel_initializer=glorot_normal(seed=1)))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(4, activation='softmax', kernel_initializer=glorot_normal(seed=1)))



model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

# loss图
plt.figure(figsize=(15, 15))
ax = plt.subplot(211)
loss = history.history['loss']
val_loss = history.history['val_loss']
epchos = range(1, len(loss) + 1)
ax.plot(epchos, loss, 'bo', label='Training loss')
ax.plot(epchos, val_loss, 'b', label='Validation loss')
ax.set_title('Training and validation loss')
ax.set_xlabel('Epchos')
ax.set_ylabel('Loss')
plt.legend()

# acc图
ax = plt.subplot(212)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epchos2 = range(1, len(acc) + 1)
ax.plot(epchos2, acc, 'bo', label='Train acc')
ax.plot(epchos2, val_acc, 'b', label='Validation acc')
ax.set_title('Train and Validation accuracy')
ax.set_xlabel('Epchos')
ax.set_ylabel('Accuracy')
plt.legend()
plt.savefig('D:\dessktop\数学建模\model loss and accuracyCNN1.png')

# testing accuracy
scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1] * 100))

scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))

# confusion matrix
plt.figure()
y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, fmt="d",
                 xticklabels=['No Sold in Past 6 months', 'Sold in past 6 months'],
                 yticklabels=['No Sold in Past 6 monthss', 'Sold in past 6 months'],
                 cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.savefig('D:\dessktop\数学建模\confusion matrixCNN1.png')

# roc图
y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.figure()
plt.plot(FPR, TPR)
plt.plot([0, 1], [0, 1], '--', color='black')  # diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig(r'D:\dessktop\数学建模\rocCNN1.png')