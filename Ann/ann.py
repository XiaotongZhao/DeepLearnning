# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Ann/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #取全表3到12列
y = dataset.iloc[:, 13].values #取13列

# Encoding categorical data
# Encoding the Independent Variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])#归一化处理
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = columnTransformer.fit_transform(X)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)#整体归一化避免出现虚拟变量陷阱以及提高训练数据的执行效率
X_test = sc.transform(X_test) #transform必须要在fit_transform后执行, 因为需要fit_transform在fit过程计算出均值μ和方差σ^2

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Inititalising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation='relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))
# #Adding the second hidden layer
classifier.add(Dense(units = 6, activation='relu'))
classifier.add(Dropout(p = 0.1))

# #Adding the output layer
classifier.add(Dense(units = 1, activation='sigmoid'))

# #Compiling the ANN
classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = new_prediction > 0.5

print(new_prediction)