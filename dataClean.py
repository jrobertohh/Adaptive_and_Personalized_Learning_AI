import pandas as pd
import os

# Construir la ruta del archivo de manera segura
base_path = '/data/'
file_name = 'conjunto_de_datos_tmodulo_enape_2021.csv'
file_path = os.path.join(base_path, file_name)

# Cargar la base de datos
df = pd.read_csv(file_path)

# Mostrar información de la base de datos original
print("Información de la base de datos original:")
print(df.info())
print("\nPrimeras filas de la base de datos original:")
print(df.head())

# Hacer una copia de la base de datos
df_cleaned = df.copy()

# Definir las columnas críticas para las cuales no queremos valores nulos
columnas_criticas = ['SEXO', 'EDAD', 'P3_1', 'PA3_1']

# Eliminar filas con valores nulos en las columnas críticas
df_cleaned.dropna(subset=columnas_criticas, inplace=True)

# Mostrar información de la base de datos limpia
print("\nInformación de la base de datos limpia:")
print(df_cleaned.info())
print("\nPrimeras filas de la base de datos limpia:")
print(df_cleaned.head())

# Guardar la base de datos limpia en un nuevo archivo CSV
cleaned_file_path = os.path.join(base_path, 'conjunto_de_datos_tmodulo_enape_2021_cleaned.csv')
df_cleaned.to_csv(cleaned_file_path, index=False)
print(f"\nLa base de datos limpia se ha guardado en: {cleaned_file_path}")