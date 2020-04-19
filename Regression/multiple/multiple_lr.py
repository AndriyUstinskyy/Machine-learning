import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

try:
    from sklearn.compose import ColumnTransformer

    onehotencoder = ct = ColumnTransformer([("state", OneHotEncoder(), [3])], remainder='passthrough')
    X = onehotencoder.fit_transform(X)
except:
    onehotencoder = OneHotEncoder(categorical_features=[3])
    X = onehotencoder.fit_transform(X).toarray()

# evitar la trampa de las variables Dummy. Para eso eliminamos una cualquiera
X = X[:, 1:]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# ajustar el modelo de Regresion lineal multiple con el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, Y_train)

# prediccion
y_pred = regression.predict(X_test)




import statsmodels.api as sm

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
SL = 0.05
# Eliminacion hacia atras eliminando paso a paso las columnas menos significativas que tienen el mayor p-valor

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()


# Eliminacion hacia atras automatica (p-valores)
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(Y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


# Eliminacion hacia atras automatica (p-valores y R-squared)
def backwardElimination2(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(Y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(Y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    print(regressor_OLS.summary())
    return x 
 
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination2(X_opt, SL)

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_Modeled, Y, test_size=0.2, random_state=1)


regression2 = LinearRegression()
regression2.fit(X_train2, Y_train2)

# prediccion
y_pred2 = regression2.predict(X_test2)
