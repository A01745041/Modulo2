# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 08:00:15 2023

@author: A0174
"""
import csv
import math
import random
import matplotlib.pyplot as plt

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    """
  Calcula la distancia euclidiana entre dos puntos.

  Args:
      point1 (list): Coordenadas del primer punto.
      point2 (list): Coordenadas del segundo punto.

  Returns:
      float: Distancia euclidiana entre los puntos.
  """
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Función para encontrar los k vecinos más cercanos
def find_neighbors(training_data, test_instance, k):
    """
Encuentra los k vecinos más cercanos a una instancia de prueba.

Args:
    training_data (list): Conjunto de datos de entrenamiento.
    test_instance (list): Instancia de prueba.
    k (int): Número de vecinos a considerar.

Returns:
    list: Lista de los k vecinos más cercanos.
"""
    distances = []
    for train_instance in training_data:
        dist = euclidean_distance(train_instance[:-1], test_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [item[0] for item in distances[:k]]
    return neighbors

# Función para predecir la etiqueta de una instancia de prueba
def predict_class(neighbors):
    """
Predice la etiqueta de una instancia de prueba basada en los vecinos cercanos.

Args:
    neighbors (list): Lista de vecinos cercanos.

Returns:
    str: Etiqueta predicha.
"""
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

# Función para calcular métricas de evaluación
def calculate_metrics(true_positive, true_negative, false_positive, false_negative):
    """
   Calcula varias métricas de evaluación.

   Args:
       true_positive (int): Verdaderos positivos.
       true_negative (int): Verdaderos negativos.
       false_positive (int): Falsos positivos.
       false_negative (int): Falsos negativos.

   Returns:
       tuple: Accuracy, precisión, recall y especificidad.
   """
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    return accuracy, precision, recall, specificity

# Cargar datos desde el archivo CSV
data = []

with open("C:/Users/A0174/OneDrive/Documentos/Breast_cancer_data.csv", "r") as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        instance = [float(value) for value in row[:-1]] + [row[-1]]
        data.append(instance)

# Lista para almacenar las precisiones en cada iteración
precisions = []
k = 5  # Número de vecinos a considerar
# Iterar sobre diferentes tamaños de datos de entrenamiento
for i in range(10, 100, 10):  # Incrementar el tamaño del conjunto de entrenamiento en pasos de 10
    """
    Este bucle itera sobre diferentes tamaños de datos de entrenamiento.
    
    El bucle varía el tamaño del conjunto de entrenamiento desde el 10% hasta el 90% del total de los datos.
    """
    split_ratio = i / 100  # Convertir el tamaño a porcentaje
    """
    split_ratio representa la proporción de datos que se utilizarán como conjunto de entrenamiento. 
    Se calcula dividiendo i (un número del rango 10, 20, ..., 90) por 100.
    Por ejemplo, cuando i es 10, split_ratio será 0.1, lo que significa que el 10% de los datos se usarán como entrenamiento.
    """

    random.shuffle(data)
    """
    Se reorganizan aleatoriamente los datos. Esto asegura que cada vez que se cambie el tamaño del conjunto de entrenamiento, 
    los datos de entrenamiento sean diferentes pero representativos del conjunto original.
    """

    split_index = int(len(data) * split_ratio)
    """
    Se calcula el índice en el que se dividirá el conjunto de datos en entrenamiento y prueba. 
    Se utiliza la proporción calculada previamente (split_ratio) para determinar el tamaño del conjunto de entrenamiento.
    """

    training_data = data[:split_index]
    test_data = data[split_index:]
    """
    Se divide el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba utilizando el índice calculado.
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    """
    Estas variables se inicializan para llevar el conteo de los resultados de la predicción del modelo.
    """

    for test_instance in test_data:
        """
        Este bucle itera sobre cada instancia en el conjunto de prueba.
        """
        neighbors = find_neighbors(training_data, test_instance[:-1], k)
        """
        Encuentra los k vecinos más cercanos a la instancia actual utilizando el conjunto de entrenamiento.
        """
        predicted_class = predict_class(neighbors)
        actual_class = test_instance[-1]
        """
        Obtiene la clase predicha por el modelo y la clase real de la instancia actual.
        """

        if actual_class == predicted_class:
            """
            Comprueba si la predicción del modelo coincide con la clase real.
            """
            if actual_class == "1":
                true_positive += 1
            else:
                true_negative += 1
            """
            Incrementa los conteos de verdaderos positivos y verdaderos negativos según corresponda.
            """
        else:
            if actual_class == "1":
                false_negative += 1
            else:
                false_positive += 1
            """
            Incrementa los conteos de falsos positivos y falsos negativos según corresponda.
            """

    # Calcular métricas
    accuracy, precision, recall, specificity = calculate_metrics(true_positive, true_negative, false_positive, false_negative)
    """
    Se calculan varias métricas de evaluación utilizando los resultados obtenidos por el modelo.
    """

    # Agregar precisión a la lista
    precisions.append(precision)
    """
    La precisión calculada se agrega a una lista para su posterior análisis y visualización.
    """

    print(f"Tamaño del conjunto de entrenamiento: {i}%")
    print ("True Positive:", true_positive)
    print ("True Negative", true_negative)
    print ("False Positive", false_positive)
    print ("False Negative", false_negative)
    print ("Accuracy;", accuracy)
    print ("Precision", precision)
    print ("Recall:", recall)
    print ("Specificity", specificity)
    print("---------")
    
    """
    Se imprime el tamaño del conjunto de entrenamiento y la precisión obtenida en esa iteración.
    """

# Graficar los resultados
plt.plot(range(10, 100, 10), precisions)
plt.xlabel('Tamaño del conjunto de entrenamiento (%)')
plt.ylabel('Precisión')
plt.title('Aprendizaje y Generalización')
plt.show()