# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:31:31 2023

@author: A0174
"""


import pandas as pd

# Cargar los datos desde el archivo CSV usando Pandas
ruta_csv = 'C:/Users/A0174/OneDrive/Documentos/healthcare-dataset-stroke-data.csv'
datos = pd.read_csv(ruta_csv)

# Dividir los datos en dos clases basadas en la etiqueta (0 y 1)
datos_clase_0 = datos[datos['stroke'] == 0]
datos_clase_1 = datos[datos['stroke'] == 1]

# Calcular las probabilidades a priori de las clases
prob_clase_0 = len(datos_clase_0) / len(datos)
prob_clase_1 = len(datos_clase_1) / len(datos)

# Función para calcular la probabilidad de una muestra dada una característica categórica
def calcular_probabilidad_categorica(valor, datos_clase):
    return len(datos_clase[datos_clase == valor]) / len(datos_clase)

# Clasificar una nueva muestra
def predecir_muestra(muestra):
    prob_clase_0_dado_muestra = prob_clase_0
    prob_clase_1_dado_muestra = prob_clase_1
    
    for feature, valor in muestra.items():
        prob_clase_0_dado_muestra *= calcular_probabilidad_categorica(valor, datos_clase_0[feature])
        prob_clase_1_dado_muestra *= calcular_probabilidad_categorica(valor, datos_clase_1[feature])
    
    if prob_clase_0_dado_muestra > prob_clase_1_dado_muestra:
        return 0
    else:
        return 1

# Casos de predicción
muestras = [
    {'gender': 'Male', 'age': 65, 'hypertension': 0, 'heart_disease': 1, 'ever_married': 'Yes'},
    {'gender': 'Female', 'age': 25, 'hypertension': 1, 'heart_disease': 0, 'ever_married': 'No'},
    {'gender': 'Female', 'age': 90, 'hypertension': 0, 'heart_disease': 1, 'ever_married': 'No'},
    {'gender': 'Male', 'age': 15, 'hypertension': 1, 'heart_disease': 0, 'ever_married': 'No'},
    {'gender': 'Female', 'age': 85, 'hypertension': 1, 'heart_disease': 1, 'ever_married': 'Yes'},
    {'gender': 'Male', 'age': 35, 'hypertension': 0, 'heart_disease': 0, 'ever_married': 'Yes'}
]

for idx, muestra in enumerate(muestras, start=1):
    prediccion = predecir_muestra(muestra)
    
    if prediccion == 0:
        print(f"Para el caso {idx}, no hay riesgo de accidente cerebrovascular.")
    else:
        print(f"Para el caso {idx}, hay riesgo de accidente cerebrovascular.")
