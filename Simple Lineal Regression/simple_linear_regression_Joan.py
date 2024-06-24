# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:40:56 2024

@author: joano
"""

#Regresion lineal simple 

#Plantilla de pre procesado

#como importar las librerias 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar el dataset

dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


#Dividir en dataset en conjunto de train y test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Crear modelo de Regresión Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)

#Predecir el conjunto de test
y_pred=regression.predict(X_test)

#Visualizar los resultados de entrenamiento
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regression.predict(X_train),color="blue")
plt.title("Sueldo vs Años de Experiencia(Conjunto de entraniemto")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo(en $)")
plt.show()


#Visualizar los resultados de entrenamiento
plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,regression.predict(X_test),color="blue")
plt.title("Sueldo vs Años de Experiencia(Conjunto de test")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo(en $)")
plt.show()
