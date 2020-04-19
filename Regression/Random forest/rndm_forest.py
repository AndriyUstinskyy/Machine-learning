import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators=300, random_state=0)
regression.fit(X, Y)


Y_pred = regression.predict(np.array([[6.5]]))


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color="red")
plt.plot(X_grid, regression.predict(X_grid), color="blue")
plt.title("Modelo de regression SVR")
plt.xlabel("Posicion del empeado")
plt.ylabel("Sueldo (en $)")
plt.show()
