"""
Abraham Gil Félix | A01750884
Inteligencia Artificial Avanzada I | Módulo 2 
"""


# Importación de librerías
from random import random
from random import seed
from math import exp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns



# Inicialización de la Red Neuronal
"""
Función inicializar_red():  permite la creación de la Red Neuronal 
especificando el número de capas de entrada, capas ocultas 
y capas de salida (parámetros).

Cada neurona en la capa oculta tiene n entradas + 1 pesos, 
uno para cada columna de entrada en un conjunto de datos y uno adicional para el bias.
Cada neurona en la capa de salida tiene un peso para cada neurona en la capa oculta.
"""
def inicializar_red(nc_entrada, nc_oculta, nc_salida):
	red = list()
	capa_oculta = [{'pesos':[random() for i in range(nc_entrada + 1)]} for i in range(nc_oculta)]
	red.append(capa_oculta)
	capa_salida = [{'pesos':[random() for i in range(nc_oculta + 1)]} for i in range(nc_salida)]
	red.append(capa_salida)
	return red


"""
Función activar(): permite calcular la activación de una neurona dada una entrada.
La activación de neuronas se calcula como la suma ponderada de las entradas.
"""
# Calcula la activación de una neurona para una entrada
def activar(pesos, entradas):
	activacion = pesos[-1]
	for i in range(len(pesos)-1):
		activacion += pesos[i] * entradas[i]
	return activacion


"""
Función transferir(): permite transferir una función de activación usando la función sigmoide.
La función sigmoide o logística es capaz de tomar cualquier valor de entrada 
y producir un número entre 0 y 1 en una curva S.
"""
# Transferir la activación de una neurona (funciones de transferencia)
def transferir(activacion):
	return 1.0 / (1.0 + exp(-activacion))


"""
Función forward_propagate(): permite la propagación hacia adelante para una fila de datos.
Devuelve las salidas de la última capa (capa de salida).
"""
# Forward propagate de una entrada a una salida de la red
def forward_propagate(red, fila):
	entradas = fila
	for capa in red:
		nuevas_entradas = []
		for neurona in capa:
			activacion = activar(neurona['pesos'], entradas)
			neurona['salida'] = transferir(activacion)
			nuevas_entradas.append(neurona['salida'])
		entradas = nuevas_entradas
	return entradas


"""
Función transferir_derivada(): permite calcular la pendiente del valor de salida de una neurona.
Utiliza la derivada de la función de transferencia sigmoide.
"""
# Calcula la derivada de la salida de una neurona
def transferir_derivada(salida):
	return salida * (1.0 - salida)


"""
Función backward_propagate_error(): implementa el algoritmo BackPropagation.

El error (en la capa de salida) para una neurona dada se puede calcular de la 
siguiente manera:
error = (salida - esperado * transferir_derivada(salida))

En la capa oculta, el error se calcula distinto.La señal de error retropropagada 
se acumula y luego se usa para determinar el error de la neurona en la capa oculta, 
de la siguiente manera:
error = (peso_k * error_j) * transferir_derivada(salida)
"""
# Backpropagate error y almacenamiento en neuronas
def backward_propagate_error(red, esperado):
	for i in reversed(range(len(red))):
		capa = red[i]
		errores = list()
		if i != (len(red)-1):
			for j in range(len(capa)):
				error = 0.0
				for neurona in red[i + 1]:
					error += (neurona['pesos'][j] * neurona['diferencia'])
				errores.append(error)
		else:
			for j in range(len(capa)):
				neurona = capa[j]
				errores.append(neurona['salida'] - esperado[j])
		for j in range(len(capa)):
			neurona = capa[j]
			neurona['diferencia'] = errores[j] * transferir_derivada(neurona['salida'])


"""
Función actualizacion_pesos(): permite actualizar los pesos después de que se calculan los errores para cada 
neurona en la red gracias al algoritmo BackPropagation.

Nota: La red se entrena utilizando descenso de gradiente estocástico, es decir, que se necesitan 
múltiples iteraciones de exponer un conjunto de datos de entrenamiento a la red y, 
para cada fila de datos, propagar hacia adelante las entradas, propagar hacia atrás el error 
y actualizar los pesos de la red neuronal.
"""
# Actualiza red pesos con error
def actualizacion_pesos(red, fila, learning_rate):
	for i in range(len(red)):
		entradas = fila[:-1]
		if i != 0:
			entradas = [neurona['salida'] for neurona in red[i - 1]]
		for neurona in red[i]:
			for j in range(len(entradas)):
				neurona['pesos'][j] -= learning_rate * neurona['diferencia'] * entradas[j]
			neurona['pesos'][-1] -= learning_rate * neurona['diferencia']


"""
Función entrenamiento_red(): implementa el entrenamiento de una red neuronal inicializada.
Parámetros: conjunto de datos de entrenamiento, tasa de aprendizaje, número de épocas
y número esperado de valores de salida.

El MSE entre la salida esperada y la salida de la red se acumula en cada época. 
Esto es útil para crear un seguimiento de cuánto está aprendiendo y mejorando la red
durante el proceso de entrenamiento.
"""
# Entrenamiento de la red para un número fijo de épocas
def entrenamiento_red(red, train_set, learning_rate, n_epocas, nc_salida):
	for epoca in range(n_epocas):
		suma_error = 0
		for fila in train_set:
			salidas = forward_propagate(red, fila)
			esperado = [0 for i in range(nc_salida)]
			esperado[fila[-1]] = 1
			suma_error += sum([(esperado[i] - salidas[i])**2 for i in range(len(esperado))])
			backward_propagate_error(red, esperado)
			actualizacion_pesos(red, fila, learning_rate)
		print('->epoca=%d, learning_rate=%.3f, error=%.3f' % (epoca, learning_rate, suma_error))
	return print('Learning rate:' + str(learning_rate) + ' | Error:' + str(suma_error))


"""
Función predecir(): permite hacer predicciones con una red neuronal entrenada. 

Propagar hacia adelante un patrón de entrada para obtener una salida es todo lo que se requiere hacer 
para que la red sea capaz de predecir.
"""
# Predecir con la red neuronal
def predecir(red, fila):
	salidas = forward_propagate(red,fila)
	return salidas.index(max(salidas))	# Función arg max.

# ------------------------------------------------------------------------------------------------------------
# Prueba del algoritmo BackPropagation para entrenamiento
# Carga de los datos
"""
Dataset: Breast Cancer Data Set

- Este es uno de los tres dominios proporcionados por el Instituto de Oncología que ha aparecido repetidamente 
en la literatura de aprendizaje automático.
- El conjunto de datos incluye 201 instancias de una clase y 85 instancias de otra clase. Las instancias se 
describen mediante 9 atributos, algunos de los cuales son lineales y otros son nominales.
"""
dataset = pd.read_csv('breast_cancer.csv')


"""
Implementación de 5-fold Cross-Validation con el conjunto 
de datos deseado. 
Se utilizó Cross-Validation con el objetivo principal de hacer varias pruebas 
variando el dataset, además de poder demostrar la efectividad y estabilidad del
algoritmo.
"""
# 5-Fold Cross-Validation
dataset = dataset.reindex(np.random.permutation(dataset.index))
dataset = dataset.reset_index(drop=True)
# folds
fold1 = dataset.loc[:(dataset.shape[0]/5)*1]                                            
fold2 = dataset.loc[(dataset.shape[0]/5)*1+1:(dataset.shape[0]/5)*2]
fold3 = dataset.loc[(dataset.shape[0]/5)*2+1:(dataset.shape[0]/5)*3]
fold4 = dataset.loc[(dataset.shape[0]/5)*3+1:(dataset.shape[0]/5)*4]
fold5 = dataset.loc[(dataset.shape[0]/5)*4+1:(dataset.shape[0])]
# train-validation
dataset1 = pd.concat([fold1, fold2, fold3, fold4])
dataset2 = pd.concat([fold1, fold2, fold3, fold5])
dataset3 = pd.concat([fold1, fold2, fold4, fold5])
dataset4 = pd.concat([fold1, fold3, fold4, fold5])
dataset5 = pd.concat([fold2, fold3, fold4, fold5])


"""
Parámetros a utilizar por la NN e inicialización de la misma.
"""
nc_entrada = len(dataset.values[0]) - 1						# Número de entradas que recibe la capa de entrada
nc_salida = len(set([fila[-1] for fila in dataset.values]))	# Número de neuronas en la capa de salida
red = inicializar_red(nc_entrada, 2, nc_salida)

print('-------------------------------------------------------------------------------------------------')
print('TRAINING')
print('-------------------------------------------------------------------------------------------------')
entrenamiento_red(red, dataset.values, 0.5, 50, nc_salida)

# Análisis del error variando el learning rate para 50 epochs
print('-------------------------------------------------------------------------------------------------')
print('TUNING HYPERPARAMETERS')
print('-------------------------------------------------------------------------------------------------')
learning_rates = [0.005, 0.05, 0.5]
for i in learning_rates:
	entrenamiento_red(red, dataset.values, i, 50, nc_salida)
	print('Finished tuning for learning_rate=' + str(i))
	print('-------------------------------------------------------------------------------------------------')
	print('-------------------------------------------------------------------------------------------------')


print('\n-------------------------------------------------------------------------------------------------')
print('VALIDATION')
print('-------------------------------------------------------------------------------------------------')
datasets = [dataset1, dataset2, dataset3, dataset4, dataset5]
for i in datasets: 
  entrenamiento_red(red, i.values, 0.05, 50, nc_salida)
  print('Finished validation of NN for fold')
  print('-------------------------------------------------------------------------------------------------')
  print('-------------------------------------------------------------------------------------------------')

# Prueba predicciones de la red neuronal
print('\n-------------------------------------------------------------------------------------------------')
print('PREDICTIONS')
print('-------------------------------------------------------------------------------------------------')
for fila in dataset.values:
  prediccion = predecir(red, fila)
print('Esperado=%d, Obtenido=%d' % (fila[-1], prediccion))