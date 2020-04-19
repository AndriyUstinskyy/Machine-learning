import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)


from sklearn.svm import SVR

regression = SVR(kernel="rbf") 
regression.fit(X, Y)


Y_pred = regression.predict(sc_X.transform(np.array([[6.5]])))
Y_pred = sc_Y.inverse_transform(Y_pred)


plt.scatter(X, Y, color="red")
plt.plot(X, regression.predict(X), color="blue")
plt.title("Modelo de regression SVR")
plt.xlabel("Posicion del empeado")
plt.ylabel("Sueldo (en $)")
plt.show()


X_normal = sc_X.inverse_transform(X)
Y_normal = sc_Y.inverse_transform(Y)