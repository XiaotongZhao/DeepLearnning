# Importing the libraries
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from keras.layers import Dense
from keras.models import Sequential
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Ann/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # 取全表3到12列
y = dataset.iloc[:, 13].values  # 取13列

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # 归一化处理
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])


columnTransformer = ColumnTransformer(
    [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = columnTransformer.fit_transform(X)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # 整体归一化避免出现虚拟变量陷阱以及提高训练数据的执行效率
# transform必须要在fit_transform后执行, 因为需要fit_transform在fit过程计算出均值μ和方差σ^2
X_test = sc.transform(X_test)

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', input_dim=11))
    classifier.add(Dense(units=6, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],'nb_epoch':[100,500],'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_accuracy)
print(best_parameters)