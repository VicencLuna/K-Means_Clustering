"""
Created on Mon Sep  2 17:34:14 2024

@author: v.luna
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns


# Cargar los datos desde el archivo CSV
df = pd.read_csv('C:\Temp/CompaniesProfiles_Scenario1.csv')

# Puedes visualizar las primeras filas para asegurarte de que los datos se han cargado correctamente
print(df.head())

df_encoded = pd.get_dummies(df, columns=['industry'], drop_first=True)

# Paso 2: Normalizar los datos (opcional, pero recomendable si las variables están en diferentes escalas)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Paso 3: Calcular WCSS para diferentes valores de K
wcss = []
K_range = range(1, 10)  # Probamos con valores de K desde 1 hasta 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)  # Inercia o WCSS para cada valor de K

# Paso 4: Graficar el método del codo
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia (WCSS)')
plt.show()

# Se sugiere k=4 o k=5 por que es donde esta la mayor diferencia.
diff_inertia = np.diff(wcss)
for i in range(len(diff_inertia)):
    print(f'De k={i+1} a k={i+2}: La inercia disminuye en {diff_inertia[i]:.2f}')

# Aplicar K-Means con k=4 (basado en el codo observado)
kmeans_final = KMeans(n_clusters=57, random_state=42)
kmeans_final.fit(df_scaled)

# Obtener los clusters asignados
clusters = kmeans_final.labels_

# Agregar los clusters al DataFrame original
df['Cluster'] = clusters
df.to_csv('C:\Temp/CompaniesProfiles_Scenario1_Cluster.csv')

# Ver el DataFrame con los clusters asignados
print(df)


plt.figure(figsize=(10, 6))
sns.stripplot(x=df['industry'], y=df['Cluster'], hue=df['Cluster'], palette='viridis', dodge=True, marker='o', size=10)
plt.title('Visualización de Clusters con una Sola Característica')
plt.xlabel('Feature1')
plt.yticks([])  # Ocultar el eje Y ya que no tiene relevancia
plt.legend(title='Cluster', loc='upper right')
plt.show()


# centroides = kmeans_final.cluster_centers_

# # Mostrar las características (centroides) de cada cluster
# for i in range(4):  # Cambia el rango según el número de clusters
#     print(f"Características del Cluster {i}:")
#     for j, col in enumerate(df.columns[:-1]):  # Excluir la columna 'Cluster'
#         print(f"{col}: {centroides[i][j]}")
#     print()  # Salto de línea entre clusters

# # Si quieres los centroides en el formato original (desescalado):
# centroides_originales = scaler.inverse_transform(centroides)
# for i in range(4):  # Cambia el rango según el número de clusters
#     print(f"Características originales del Cluster {i}:")
#     for j, col in enumerate(df.columns[:-1]):  # Excluir la columna 'Cluster'
#         print(f"{col}: {centroides_originales[i][j]}")
#     print()  # Salto de línea entre clusters


# Agrupar por 'industria' y 'cluster' y contar el número de clientes
df_count = df.groupby(['industry', 'Cluster']).size().reset_index(name='numero_clientes')

# Ordenar las industrias por el número de clusters
cluster_count = df_count.sort_values('Cluster', ascending=False)
ordered_industries = cluster_count['industry'].tolist()

# Aplicar el orden de industrias al DataFrame df_count
df_count['industry'] = pd.Categorical(df_count['industry'], categories=ordered_industries, ordered=True)
df_count = df_count.sort_values('industry')

# Mostrar el DataFrame resultante
print(df_count)

# Crear el gráfico de burbujas
plt.figure(figsize=(12, 8))

# Crear el gráfico de burbujas con seaborn
sns.scatterplot(
    x='industry', 
    y='Cluster', 
    size='numero_clientes', 
    sizes=(100, 1000),  # Ajusta los tamaños mínimo y máximo de las burbujas
    hue='Cluster', 
    data=df_count, 
    palette='viridis', 
    legend=True,  # Puedes ajustar la leyenda según sea necesario
    alpha=0.6
)

# Títulos y etiquetas
plt.title('Gráfico de Burbujas: Cluster por Industria y Número de Clientes')
plt.xlabel('Industria')
plt.ylabel('Cluster')
plt.grid(True)
