import numpy as np
import pandas as pd

# Cargar el archivo CSV proporcionado por el usuario
file_path = r'C:\Users\s2dan\OneDrive\Documentos\WorkSpace\PrimerParcia_IA\ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# Normalización manual (min-max normalization) para las columnas numéricas
def normalize_column(column):
    return (column - np.min(column)) / (np.max(column) - np.min(column))

# Seleccionar columnas numéricas para normalización
numerical_columns = ['Age', 'Height', 'Weight', 'CH2O', 'FAF', 'TUE']
normalized_data = data[numerical_columns].apply(normalize_column)

# Simular algunos coeficientes de regresión (por ejemplo, obtenidos de un modelo)
# Nota: Estos coeficientes normalmente son los pesos aprendidos de un modelo de regresión
coefficients = np.random.randn(len(numerical_columns))  # Coeficientes aleatorios

# Definir el factor de regularización (lambda)
lambda_reg = 0.1

# Calcular la penalización L1
def l1_penalty(coefficients, lambda_reg):
    return lambda_reg * np.sum(np.abs(coefficients))

# Calcular la penalización L2
def l2_penalty(coefficients, lambda_reg):
    return lambda_reg * np.sum(coefficients ** 2)

# Calcular las penalizaciones
l1 = l1_penalty(coefficients, lambda_reg)
l2 = l2_penalty(coefficients, lambda_reg)

# Mostrar los resultados
print(f"Coeficientes: {coefficients}")
print(f"Penalización L1: {l1}")
print(f"Penalización L2: {l2}")
