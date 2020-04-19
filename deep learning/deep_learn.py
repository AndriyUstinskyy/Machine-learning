import numpy as np
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

clf = Sequential()
clf.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu" , input_dim = 11))

clf.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu" ))

clf.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid" ))

#compilar la red

clf.compile(optimizer = "adam", loss ="binary_crossentropy", metrics =["accuracy"])

#entrenamiento

clf.fit(X_train, Y_train, batch_size = 10, epochs = 100)


y_pred = clf.predict(X_test)

y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

result = (cm[0,0] + cm[1,1])/2000

print(result)






