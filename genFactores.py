# Código completo para generar Análisis Factorial y Modelo Gráfico
# Autor: Asistente Experto para Revisor Q1

import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import numpy as np
import graphviz
import os
import warnings

# Ignorar advertencias de versiones futuras para mantener la salida limpia
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y CARGA DE DATOS
# ==========================================

# Ruta relativa correcta basada en tu estructura de carpetas
base_path = os.path.join('data')
file_name = 'conjunto_de_datos_tmodulo_enape_2021.csv'
file_path = os.path.join(base_path, file_name)

print(f"--> Cargando archivo desde: {file_path}")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"No se encontró el archivo en: {file_path}. Verifica la carpeta 'data'.")

# Cargar la base de datos
df = pd.read_csv(file_path)

# ==========================================
# 2. PREPROCESAMIENTO Y LIMPIEZA
# ==========================================

# Filtrar datos para nivel superior y edad >= 18
df_filtered = df[(df['PA3_3_NIVEL'] >= 8) & (df['EDAD'] >= 18)].copy()

# LISTA DE COLUMNAS A EXCLUIR
# Eliminamos identificadores, datos demográficos y factores de expansión
# para dejar solo las variables que miden constructos latentes.
cols_to_exclude = [
    'FOLIO', 'N_REN', 'SEXO', 'EDAD', 'FACTOR', 'P3_1', 'ENT', 'MUN',
    'FILTRO_A', 'FILTRO_B', 'FILTRO_D', 'NIVEL_A', 'GRADO_A', 'NIVEL_B',
    'GRADO_B', 'ESC'
]

# Seleccionar solo numéricas
numeric_df = df_filtered.select_dtypes(include=[np.number])

# Eliminar columnas explícitas de la lista de exclusión si existen
cols_present = [c for c in cols_to_exclude if c in numeric_df.columns]
numeric_df = numeric_df.drop(columns=cols_present)

# Eliminar columnas con más del 50% de NaNs
threshold = len(numeric_df) * 0.5
numeric_df = numeric_df.dropna(thresh=threshold, axis=1)

# Reemplazar infinitos y rellenar nulos
numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_df.fillna(0, inplace=True)

# Eliminar columnas con varianza cero (constantes)
variance = numeric_df.var()
zero_variance_columns = variance[variance == 0].index.tolist()
numeric_df = numeric_df.drop(columns=zero_variance_columns)

print(f"--> Dimensiones finales para análisis: {numeric_df.shape} (Filas, Variables)")

# Normalizar datos (Z-score)
normalized_df = numeric_df.apply(lambda x: (x - x.mean()) / x.std())

# ==========================================
# 3. VALIDACIÓN ESTADÍSTICA (KMO Y BARTLETT)
# ==========================================
print("\n" + "=" * 40)
print("RESULTADOS DE ADECUACIÓN MUESTRAL")
print("=" * 40)

# Prueba de Esfericidad de Bartlett
try:
    chi_square_value, p_value = calculate_bartlett_sphericity(normalized_df)
    print(f"Prueba de Bartlett:\n  Chi-cuadrado: {chi_square_value:.2f}")
    print(f"  P-valor: {p_value} (Significativo si < 0.05)")
except Exception as e:
    print(f"Error en Bartlett: {e}")

# Índice KMO (Kaiser-Meyer-Olkin)
try:
    kmo_all, kmo_model = calculate_kmo(normalized_df)
    print(f"Índice KMO Global: {kmo_model:.4f}")
    if kmo_model > 0.8:
        print("  Interpretación: MERITORIO/EXCELENTE para Análisis Factorial.")
    elif kmo_model > 0.6:
        print("  Interpretación: MEDIOCRE pero aceptable.")
    else:
        print("  Interpretación: INADECUADO.")
except Exception as e:
    print(f"Error en KMO: {e}")

# ==========================================
# 4. ANÁLISIS FACTORIAL (EFA)
# ==========================================

# Ajustar el modelo factorial
# Usamos 'minres' (Minimum Residual) que es robusto y común en EFA
fa = FactorAnalyzer(n_factors=4, rotation='varimax', method='minres')
fa.fit(normalized_df)

# Obtener Cargas Factoriales
loadings = pd.DataFrame(fa.loadings_, index=normalized_df.columns, columns=['F1', 'F2', 'F3', 'F4'])

print("\n" + "=" * 40)
print("CARGAS FACTORIALES (Variables > 0.4)")
print("=" * 40)
# Mostrar solo cargas relevantes para facilitar lectura
print(loadings[loadings.abs() > 0.4].fillna(''))

# ==========================================
# 5. GENERACIÓN DE GRÁFICO (GRAPHVIZ)
# ==========================================

try:
    dot = graphviz.Digraph(comment='Modelo Factores')

    # Configuración estética del grafo
    dot.attr(rankdir='LR', size='10')
    dot.attr('node', shape='ellipse', style='filled', color='lightblue')

    # Nodos Factores
    dot.node('F1', 'Factor 1\nImpacto Vida Diaria')
    dot.node('F2', 'Factor 2\nAcceso Educación')
    dot.node('F3', 'Factor 3\nDesempeño Educativo')
    dot.node('F4', 'Factor 4\nPercepción Educación')

    # Nodos Variables (Cajas)
    dot.attr('node', shape='box', style='solid', color='black')

    # Mapeo manual de variables teóricas para el diagrama
    # (Asegúrate de que estas columnas existan en tu CSV filtrado)
    variables_map = {
        'F1': ['PB3_6', 'PB3_8', 'PB3_9_1', 'PB3_9_2'],
        'F2': ['PA3_3_SEMESTRE', 'PA3_6'],
        'F3': ['PA3_7_1', 'PA3_7_2', 'PA3_7_3'],
        'F4': ['PA3_8_1', 'PA3_8_2', 'PA3_8_3']
    }

    for factor, vars_list in variables_map.items():
        for var in vars_list:
            # Solo agregar al gráfico si la variable sobrevivió a la limpieza
            if var in normalized_df.columns:
                dot.node(var, var)
                dot.edge(factor, var)

    # Relaciones entre factores (opcional)
    dot.edge('F1', 'F4', dir='both', style='dashed')
    dot.edge('F2', 'F3', dir='both', style='dashed')

    # Guardar
    output_path = 'img/modelo_factores'
    if not os.path.exists('img'):
        os.makedirs('img')

    dot.render(output_path, format='png', cleanup=True)
    print(f"\n--> Gráfico generado exitosamente en: {output_path}.png")

except Exception as e:
    print(f"\nError generando gráfico Graphviz: {e}")
    print("Asegúrate de tener instalado Graphviz en el sistema (brew install graphviz).")