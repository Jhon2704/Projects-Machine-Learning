# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:54:10 2024

@author: joano
"""


#librerias 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importar el dataset

dataset = pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


#Evitar la trampa de la variable ficticia
X=X[:,1:]

#Dividir en dataset en conjunto de train y test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)

#Prediccion de los resultados en el conjunto de testing
y_pred=regression.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Ploteo de los resultados
plt.scatter(range(len(y_test)), y_test, color='red', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Profit')
plt.legend()
plt.show()
