# Modulo2
Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)

# Instrucciones
1) Descargar ambos archivos adjuntos.
2) Abrir el código en el compilador deseado.
3) En el código encontrarás la línea 52
   with open("C:/Users/A0174/OneDrive/Documentos/Breast_cancer_data.csv", "r") as csvfile:
4) Cambiar el path del csv descargado
5) Correr el código normalmente

# KNN- Explicación
K-Nearest Neighbors (KNN) es un algoritmo de aprendizaje automático supervisado que se utiliza para clasificación y regresión. En este algoritmo, una versión desconocida se clasifica según sus vecinos más cercanos en el espacio de características. La idea principal detrás de KNN es que las versiones similares tienden a agruparse en el mismo espacio, por lo que si conocemos las etiquetas de las versiones vecinas, podemos predecir la etiqueta de una versión no conocida.

# Documentación General 
En este código, se hicieron diferentes funciones para poder generar un algoritmo de KNN funcional. Empezamos con aargar los datos, donde cargamos los datos desde un archivo CSV. Cada fila del archivo representa una instancia con características y una etiqueta (clase). A partir de esto, dividimos los datos en dos conjuntos: uno para entrenamiento y otro para prueba. En este caso, el 80% de los datos se utilizan para entrenar el modelo y el 20% restante se utilizan para probar el modelo.

En el ládo matemático, calculamos la distancia euclideana por medio de la creación y definición de una función denominada euclidean_distance que calcula la distancia euclidiana entre dos puntos (instancias) en un espacio de características. Después de esto, se defininió la función find_neighbors que encuentra los K vecinos más cercanos para una instancia de prueba en función de la distancia euclidiana. Finalmente, creamos la función predict_class toma los vecinos más cercanos y predice la etiqueta para la instancia de prueba basada en las etiquetas de los vecinos.

Para la parte de la evaluación del modelo, creamos la función calculate_metrics la calcula métricas de evaluación como precisión, exactitud, recall y especificidad, utilizando los valores de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos. De esta manera evaluamos el modelo usando un bucle donde iteramos a través de las instancias de prueba. Para cada instancia de prueba, encontramos sus vecinos más cercanos, predecimos su etiqueta y comparamos con la etiqueta real para actualizar las métricas.

Y para concluir, calculamos las métricas finales y mostramos los resultados, incluyendo verdaderos positivos, verdaderos negativos, falsos positivos, falsos negativos, precisión, exactitud, recall y especificidad. Esto, además, se ve graficado con matplolib para poder saber si el modelo está aprendiendo y si generaliza. 

