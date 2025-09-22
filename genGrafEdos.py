# Anexo C. Código para generar gráfico por estados

import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
file_path = '/mnt/c/Users/Dell Precision 7670/Downloads/conjunto_de_datos_enape_2021_csv/conjunto_de_datos_tmodulo_enape_2021/conjunto_de_datos/conjunto_de_datos_tmodulo_enape_2021.csv'
df = pd.read_csv(file_path)

# Filtrar datos para nivel superior y edad >= 18
df_filtered = df[(df['PA3_3_NIVEL'] >= 8) & (df['EDAD'] >= 18)]

# Contar número de personas por estado
estado_counts = df_filtered['ENT'].value_counts().sort_index()

# Crear el gráfico
plt.figure(figsize=(12, 8))
bars = plt.bar(estado_counts.index, estado_counts.values, color='skyblue')
plt.xlabel('Estado')
plt.ylabel('Número de personas')
plt.title('Número de personas de 18 años o más que estudian o estudiaron en la educación superior por estado')
plt.xticks(rotation=90)

# Añadir el número de personas encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

plt.tight_layout()

# Guardar el gráfico como imagen
plt.savefig('img/personas_por_estado.png')
plt.show()