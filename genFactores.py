# Código para generar el modelo de factores gráficos

import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
import numpy as np
import graphviz
import os

# Construir la ruta del archivo de manera segura
base_path = '/data/'
file_name = 'conjunto_de_datos_tmodulo_enape_2021.csv'
file_path = os.path.join(base_path, file_name)

# Cargar la base de datos
df = pd.read_csv(file_path)

# Filtrar datos para nivel superior y edad >= 18
df_filtered = df[(df['PA3_3_NIVEL'] >= 8) & (df['EDAD'] >= 18)]

# Seleccionar solo las columnas numéricas para el análisis factorial
numeric_df = df_filtered.select_dtypes(include=[np.number])

# Eliminar columnas con más del 50% de NaNs
threshold = len(numeric_df) * 0.5
numeric_df = numeric_df.dropna(thresh=threshold, axis=1)

# Reemplazar valores infinitos por NaN
numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Verificar qué columnas aún contienen NaNs
columns_with_nans = numeric_df.columns[numeric_df.isna().any()].tolist()
print("Columnas con NaN antes de rellenar:", columns_with_nans)

# Rellenar NaNs con ceros
numeric_df.fillna(0, inplace=True)

# Verificar nuevamente qué columnas aún contienen NaNs
columns_with_nans_after = numeric_df.columns[numeric_df.isna().any()].tolist()
print("Columnas con NaN después de rellenar:", columns_with_nans_after)

# Verificar que no hay NaNs o infinitos en el DataFrame
print("Número de NaNs después de rellenar con ceros:", numeric_df.isnull().sum().sum())
print("Número de infinitos después de rellenar con ceros:", np.isinf(numeric_df).sum().sum())

# Eliminar columnas con varianza cero
variance = numeric_df.var()
zero_variance_columns = variance[variance == 0].index.tolist()
numeric_df = numeric_df.drop(columns=zero_variance_columns)
print("Columnas con varianza cero:", zero_variance_columns)

# Asegurarse de que todavía tenemos datos después de eliminar filas con NaNs
print(f"Dimensiones del DataFrame después de limpiar: {numeric_df.shape}")

# Normalizar el DataFrame
normalized_df = numeric_df.apply(lambda x: (x - x.mean()) / x.std())

# Reemplazar posibles NaNs resultantes de la normalización con ceros
normalized_df.replace([np.inf, -np.inf], np.nan, inplace=True)
normalized_df.fillna(0, inplace=True)

# Verificar nuevamente que no hay NaNs ni infinitos después de la normalización
print("Número de NaNs después de normalizar:", normalized_df.isnull().sum().sum())
print("Número de infinitos después de normalizar:", np.isinf(normalized_df).sum().sum())

# Verificar si hay algún valor no numérico
if normalized_df.applymap(np.isreal).all().all():
    print("Todos los valores en el DataFrame normalizado son numéricos.")
else:
    print("Existen valores no numéricos en el DataFrame normalizado.")

# Ajustar el modelo factorial
fa = FactorAnalyzer(n_factors=4, rotation='varimax')
fa.fit(normalized_df)

# Obtener los factores
factor_scores = fa.transform(normalized_df)

# Crear un DataFrame con los factores
factor_df = pd.DataFrame(factor_scores, columns=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4'])

# Crear el modelo gráfico con Graphviz
dot = graphviz.Digraph(comment='Modelo Factores')

# Añadir nodos para los factores
dot.node('F1', 'Factor 1\n(Impacto de la educación en la vida diaria)')
dot.node('F2', 'Factor 2\n(Acceso a la educación)')
dot.node('F3', 'Factor 3\n(Desempeño educativo)')
dot.node('F4', 'Factor 4\n(Percepción de la educación)')

# Añadir nodos para las variables observadas
variables = {
    'F1': ['PB3_6', 'PB3_8', 'PB3_9_1', 'PB3_9_2'],
    'F2': ['PA3_3_SEMESTRE', 'PA3_6'],
    'F3': ['PA3_7_1', 'PA3_7_2', 'PA3_7_3'],
    'F4': ['PA3_8_1', 'PA3_8_2', 'PA3_8_3']
}

for factor, vars in variables.items():
    for var in vars:
        dot.node(var, var)
        dot.edge(factor, var)

# Añadir flechas bidireccionales entre los factores especificados
dot.edge('F1', 'F4', dir='both')
dot.edge('F2', 'F3', dir='both')

# Renderizar el modelo gráfico
dot.render('img/modelo_factores', format='png')

print("Gráficos y matriz de correlación guardados en la carpeta 'img'.")