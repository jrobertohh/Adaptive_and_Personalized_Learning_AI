import pandas as pd
import os

# Construir la ruta del archivo de manera segura
base_path = '/data/'
file_name = 'conjunto_de_datos_tmodulo_enape_2021_cleaned.csv'
file_path = os.path.join(base_path, file_name)

# Cargar la base de datos limpia
df_cleaned = pd.read_csv(file_path)

# Filtrar datos para personas de 18 años en adelante que estén cursando la educación superior o la hayan cursado
df_filtered = df_cleaned[(df_cleaned['EDAD'] >= 18) & (df_cleaned['NIVEL_A'] == 9)]

# Calcular las frecuencias de hombres y mujeres
frecuencia_sexo = df_filtered['SEXO'].value_counts()

# Mostrar el resultado
print("Cantidad de hombres y mujeres de 18 años en adelante que están cursando o han cursado la educación superior:")
print(frecuencia_sexo)