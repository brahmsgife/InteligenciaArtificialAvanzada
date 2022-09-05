"""
Abraham Gil Félix | A01750884
Inteligencia Artificial Avanzada I | Módulo 2 
"""


# Importación de librerías
from random import random
from random import seed
from math import exp


# Inicialización de la Red Neuronal
"""
Función que permite que la creación de la Red Neuronal 
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
Función que permite calcular la activación de una neurona dada una entrada.
La activación de neuronas se calcula como la suma ponderada de las entradas.
"""
# Calcula la activación de una neurona para una entrada
def activar(pesos, entradas):
	activacion = pesos[-1]
	for i in range(len(pesos)-1):
		activacion += pesos[i] * entradas[i]
	return activacion


"""
Función que permite transferir una función de activación usando la función sigmoide.
La función sigmoide o logística es capaz de tomar cualquier valor de entrada 
y producir un número entre 0 y 1 en una curva S.
"""
# Transferir la activación de una neurona (funciones de transferencia)
def transferir(activacion):
	return 1.0 / (1.0 + exp(-activacion))


"""
Función que permite la propagación hacia adelante para una fila de datos.
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
Función que permite calcular la pendiente del valor de salida de una neurona.
Utiliza la derivada de la función de transferencia sigmoide.
"""
# Calcula la derivada de la salida de una neurona
def transferir_derivada(salida):
	return salida * (1.0 - salida)


"""
Función que implementa el algoritmo BackPropagation.

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
Función que permite actualizar los pesos después de que se calculan los errores para cada 
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
Función que implementa el entrenamiento de una red neuronal inicializada.
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


"""
Función que permite hacer predicciones con una red neuronal entrenada. 

Propagar hacia adelante un patrón de entrada para obtener una salida es todo lo que se requiere hacer 
para que la red sea capaz de predecir.
"""
# Predecir con la red neuronal
def predecir(red, fila):
	salidas = forward_propagate(red,fila)
	return salidas.index(max(salidas))	# Función arg max.



# Prueba del algoritmo BackPropagation para entrenamiento
seed(1)
dataset = [[2.7811,2.5505,0],
	[1.4655,2.3621,0],
	[3.3966,4.4003,0],
	[1.3881,1.8502,0],
	[3.0641,3.00530,0],
	[7.6275,2.7593,1],
	[5.3324,2.0886,1],
	[6.9226,1.7711,1],
	[8.6754,-0.2421,1],
	[7.6738,3.5086,1]]

nc_entrada = len(dataset[0]) - 1						# Número de entradas que recibe la capa de entrada
nc_salida = len(set([fila[-1] for fila in dataset]))	# Número de neuronas en la capa de salida
red = inicializar_red(nc_entrada, 2, nc_salida)

print('-------------------------------------------------------------------------------------------------')
print('TRAINING')
print('-------------------------------------------------------------------------------------------------')
entrenamiento_red(red, dataset, 0.5, 50, nc_salida)


# Prueba predicciones de la red neuronal
print('\n-------------------------------------------------------------------------------------------------')
print('PREDICTIONS')
print('-------------------------------------------------------------------------------------------------')
for fila in dataset:
	prediccion = predecir(red, fila)
	print('Esperado=%d, Obtenido=%d' % (fila[-1], prediccion))