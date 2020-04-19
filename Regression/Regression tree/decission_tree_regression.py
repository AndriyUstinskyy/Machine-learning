import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values

from sklearn.tree import DecisionTreeRegressor

regression = DecisionTreeRegressor(random_state=0)
regression.fit(X, Y)

Y_pred = regression.predict(np.array([[6.5]]))

plt.scatter(X, Y, color="red")
plt.plot(X, regression.predict(X), color="blue")
plt.title("Modelo de regression SVR")
plt.xlabel("Posicion del empeado")
plt.ylabel("Sueldo (en $)")
plt.show()
