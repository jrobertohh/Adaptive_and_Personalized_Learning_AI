# Código usado para generar el análisis de Correlación y Varianzas, entre otros

import pandas as pd
import matplotlib.pyplot as plt
import os
from factor_analyzer import FactorAnalyzer
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Construir la ruta del archivo de manera segura
base_path = '/mnt/c/Users/Dell Precision 7670/Downloads/conjunto_de_datos_enape_2021_csv/conjunto_de_datos_tmodulo_enape_2021/conjunto_de_datos'
file_name = 'conjunto_de_datos_tmodulo_enape_2021.csv'
file_path = os.path.join(base_path, file_name)

# Cargar la base de datos
df = pd.read_csv(file_path)

# Filtrar datos para nivel superior y edad >= 18
df_filtered = df[(df['PA3_3_NIVEL'] >= 8) & (df['EDAD'] >= 18)]

# Contar hombres y mujeres
sexo_counts = df_filtered['SEXO'].value_counts()
hombres = sexo_counts.get(1, 0)
mujeres = sexo_counts.get(2, 0)

# Imprimir la distribución de hombres y mujeres
print(f"Total de hombres (18 años o más): {hombres}")
print(f"Total de mujeres (18 años o más): {mujeres}")

# Crear el folder 'img' si no existe
if not os.path.exists('img'):
    os.makedirs('img')

# Gráfico 1: Distribución por Sexo
labels = ['Hombres', 'Mujeres']
sizes = [hombres, mujeres]
colors = ['#ff9999','#66b3ff']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

# Guardar el gráfico como imagen
plt.savefig('img/distribucion_sexo.png')
plt.close()

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

# Asegurarse de que todavía tenemos datos después de eliminar filas con NaNs
print(f"Dimensiones del DataFrame después de limpiar: {numeric_df.shape}")

# Eliminar columnas con varianza cero
variance = numeric_df.var()
columns_with_zero_variance = variance[variance == 0].index.tolist()
numeric_df = numeric_df.drop(columns=columns_with_zero_variance)
print(f"Columnas con varianza cero: {columns_with_zero_variance}")

# Normalizar el DataFrame utilizando StandardScaler
scaler = StandardScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

# Verificar que no hay NaNs ni infinitos después de la normalización
print("Número de NaNs después de normalizar:", normalized_df.isnull().sum().sum())
print("Número de infinitos después de normalizar:", np.isinf(normalized_df).sum().sum())

# Verificar si hay algún valor no numérico
if normalized_df.applymap(np.isreal).all().all():
    print("Todos los valores en el DataFrame normalizado son numéricos.")
else:
    print("Existen valores no numéricos en el DataFrame normalizado.")

# Verificar nuevamente si hay NaNs o Inf antes de ajustar el modelo
if normalized_df.isnull().sum().sum() > 0 or np.isinf(normalized_df).sum().sum() > 0:
    print("Aún hay NaNs o valores infinitos en el DataFrame. Revisar datos.")
else:
    # Ajustar el modelo factorial con 4 factores en lugar de 2
    fa = FactorAnalyzer(n_factors=4, rotation='varimax')
    fa.fit(normalized_df)

    # Obtener los factores
    factor_scores = fa.transform(normalized_df)

    # Crear un DataFrame con los factores
    factor_df = pd.DataFrame(factor_scores, columns=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4'])

    # Calcular las correlaciones
    correlation_matrix = factor_df.corr()

    # Guardar la matriz de correlación en un archivo CSV
    correlation_matrix.to_csv('img/correlation_matrix.csv')

    # Gráfico 2: Varianza explicada por los factores
    factors = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']
    variance_explained = fa.get_factor_variance()[0]

    fig2, ax2 = plt.subplots()
    ax2.bar(factors, variance_explained, color=['blue', 'orange', 'green', 'purple'])
    ax2.set_xlabel('Factores')
    ax2.set_ylabel('Varianza Explicada')
    ax2.set_title('Varianza Explicada por Cada Factor')

    # Guardar el gráfico como imagen
    plt.savefig('img/varianza_factores.png')
    plt.close()

    # Gráfico 3: Matriz de correlación
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title('Matriz de Correlación de los Factores')

    # Guardar el gráfico como imagen
    plt.savefig('img/matriz_correlacion.png')
    plt.close()

print("Gráficos y matriz de correlación guardados en la carpeta 'img'.")