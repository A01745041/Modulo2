# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:44:08 2023

@author: A0174
"""

import csv
import math
import random

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Función para encontrar los k vecinos más cercanos
def find_neighbors(training_data, test_instance, k):
    distances = []
    for train_instance in training_data:
        dist = euclidean_distance(train_instance[:-1], test_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [item[0] for item in distances[:k]]
    return neighbors

# Función para predecir la etiqueta de una instancia de prueba
def predict_class(neighbors):
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

# Dividir los datos en entrenamiento y prueba
random.shuffle(data)
split_ratio = 0.8  # Porcentaje de datos para entrenamiento

split_index = int(len(data) * split_ratio)
training_data = data[:split_index]
test_data = data[split_index:]

# Asumiendo que test_data es una instancia de prueba con características
k = 5  # Número de vecinos a considerar

# Calcular métricas
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# Iterar a través de las instancias de prueba
for test_instance in test_data:
    neighbors = find_neighbors(training_data, test_instance[:-1], k)
    predicted_class = predict_class(neighbors)
    actual_class = test_instance[-1]  # Etiqueta real

    # Actualizar métricas
    if actual_class == predicted_class:
        if actual_class == "1":  # Positive class
            true_positive += 1
        else:
            true_negative += 1
    else:
        if actual_class == "1":  # Positive class
            false_negative += 1
        else:
            false_positive += 1

# Calcular métricas y mostrar resultados
accuracy, precision, recall, specificity = calculate_metrics(true_positive, true_negative, false_positive, false_negative)

print("True Positive:", true_positive)
print("True Negative:", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)