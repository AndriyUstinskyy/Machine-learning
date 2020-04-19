
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1 / 3, random_state=0)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(X_train, Y_train)

y_predit = regression.predict(X_test)

# visualizar

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")

plt.title("Sueldo vs Anos de experiencia(conjunto de entrenamiento)")
plt.xlabel("Años")
plt.ylabel("Sueldo")
plt.show()

plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")

plt.title("Sueldo vs Años de experiencia(conjunto de entrenamiento)")
plt.xlabel("Años")
plt.ylabel("Sueldo")
plt.show()
