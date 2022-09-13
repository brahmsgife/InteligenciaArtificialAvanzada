"""
Abraham Gil Félix | A01750884
"""

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score,  confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV



# Carga de los datos
"""
Dataset: Breast Cancer Data Set

- Este es uno de los tres dominios proporcionados por el Instituto de Oncología que ha aparecido repetidamente 
en la literatura de aprendizaje automático.
- El conjunto de datos incluye 201 instancias de una clase y 85 instancias de otra clase. Las instancias se 
describen mediante 9 atributos, algunos de los cuales son lineales y otros son nominales.
"""
data = pd.read_csv('breast-cancer.data')


# Preprocesamiento y transformación de los datos
"""
- Imputación del nombre a cada variable del dataset.
"""
data.columns = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 
                'breast', 'breast-quad', 'irradiat']


"""
- Reemplzao de valores faltantes (representados con '?') en las variables "node-caps" y 
"breast-quad" del dataset.
"""
data['node-caps'].replace(to_replace='?', value=np.NaN, inplace=True)
data['node-caps'].fillna('no', inplace=True)
data['breast-quad'].replace(to_replace='?', value=np.NaN, inplace=True)
data['breast-quad'].fillna('left_low', inplace=True)


"""
Función transform_data(): que transforma los datos de cada columna con ayuda del método LabelEncoder para que todas 
y cada una de las variables sean de tipo numérico.  
"""
def transform_data(dataset):
  labelencoder = LabelEncoder()
  for i in dataset.columns:
    dataset[i] = labelencoder.fit_transform(dataset[i])
    dataset[i].astype(int)
  return dataset

data = transform_data(data)


# Partición del conjunto de datos
"""
- Separación de las variables independientes y la variable dependiente del conjunto de datos.

"""
X = data.drop('class', axis=1)
y = data['class']

"""
- Partición de los datos en conjuntos para entrenamiento y prueba del modelo.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)


# Modelación - Creación Random Forest
"""
Función entrenar_modelo(): permite crear como modelo Random Forest con los siguientes parámetros
- n_estimators = 1000
- criterion = entropy
- max_depth = 3.0
- max_features = 5
- class_weight = None
Estos fueron los resultados que obtuvo el algoritmo GridSearch para el tuneo de hiperparámetros.
""" 
def entrenar_modelo(X_train, X_test, y_train, y_test):
    rfc_base = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=3, max_features=5, 
                                        bootstrap=False, random_state=42)
    rfc_base.fit(X_train, y_train)
    return rfc_base

"""
Función cv_entrenamiento(): que permite realizar 5-fold Cross-Validation con los conjuntos de entrenamiento.
"""
def cv_entrenamiento(X_train, y_train):
  cv_scores_train = cross_val_score(rfc, X_train, y_train, cv=5)
  print('5-fold Cross-Validation | Train')
  print('------------------------------------------------------------------------------')
  print(cv_scores_train)
  print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_train)))
  print('------------------------------------------------------------------------------')

"""
Función cv_prueba(): permite realizar 5-fold Cross-Validation con los conjuntos de prueba.
"""
def cv_prueba(X_test, y_test):
  cv_scores_test = cross_val_score(rfc, X_test, y_test, cv=5)
  print('5-fold Cross-Validation | Test')
  print('------------------------------------------------------------------------------')
  print(cv_scores_test)
  print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_test)))
  print('------------------------------------------------------------------------------')

"""
Función cv_resultados(): crea una gráfica de barras con el accuracy obtenido en cada validación cruzada 
con el conjunto dedatos en entrenamiento y prueba.
"""
def cv_resultados(x_label, y_label, plot_title, train_data, val_data):
  plt.figure(figsize=(12,6))
  labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
  X_axis = np.arange(len(labels))
  ax = plt.gca()
  plt.ylim(0.40000, 1)
  plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
  plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
  plt.title(plot_title, fontsize=30)
  plt.xticks(X_axis, labels)
  plt.xlabel(x_label, fontsize=14)
  plt.ylabel(y_label, fontsize=14)
  plt.legend()
  plt.grid(True)
  plt.show()

"""
Función resultados(): permite visualizar los resultados obtenidos por el modelo (Random Forest) en
la etapa de prueba.  
"""
labels=['no eventos recurrentes', 'eventos recurrentes']
def resultados(y_test, y_pred):
  conf_matrix = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(12, 12))
  sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
  plt.title("Confusion matrix")
  plt.ylabel('True class')
  plt.xlabel('Predicted class')
  plt.show()
  print('Results - Random Forest Classifier')
  print('------------------------------------------------------------------------------')
  print('Confusion Matrix: \n {} \n Accuracy Score: \n {} \n Recall Score: \n {}'.format(
  confusion_matrix(y_test, y_pred),
  accuracy_score(y_test, y_pred), 
  recall_score(y_test, y_pred)))
  print('------------------------------------------------------------------------------')
  print (classification_report(y_test, y_pred))


# Random Forest Classifier
"""
- Entrenamiento y evaluación del modelo creado (Random Forest).
"""
rfc = entrenar_modelo(X_train, X_test, y_train, y_test)
cv_entrenamiento(X_train, y_train)
cv_prueba(X_test, y_test)
cv_results = cross_validate(rfc, X=X, y=y, cv=5, scoring='accuracy', return_train_score=True)
cv_resultados("Random Forest","Accuracy", "Accuracy scores in 5 Folds",
            cv_results['train_score'],
            cv_results['test_score'])
y_pred = rfc.predict(X_test)
resultados(y_test, y_pred)

"""
- Predicciones
"""
print('\n-------------------------------------------------------------------------------------------------')
print('PREDICTIONS')
print('-------------------------------------------------------------------------------------------------')
print('\n class 0 no-recurrence-events')
print('\n class 1 recurrence-events')
print('-------------------------------------------------------------------------------------------------')
d = {'age': [2], 'menopause': [2], 'tumor-size': [7], 'inv-nodes': [0], 'node-caps': [0], 'deg-malig': [0], 
     'breast': [0], 'breast-quad':[1], 'irradiat':[0]}
print('\n Query:', d)
data_pred = pd.DataFrame(data=d)
pred =  rfc.predict(data_pred)
print("\n Prediction of class for query" , pred)