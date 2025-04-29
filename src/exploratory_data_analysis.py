import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Obtener la ruta absoluta al directorio raíz del proyecto
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root_dir, 'data', 'elliptic_bitcoin_dataset')
results_dir = os.path.join(root_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# 1. Carga de datos
print("Cargando datos...")
df_classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))
df_edgelist = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))
df_feats = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None)
df_feats.columns = ['txid'] + [f"feat_{i}" for i in range(1, df_feats.shape[1])]

# 2. Análisis básico de los datos
print("\n=== ANÁLISIS BÁSICO DE LOS DATOS ===")
print(f"Dimensiones de df_classes: {df_classes.shape}")
print(f"Dimensiones de df_edgelist: {df_edgelist.shape}")
print(f"Dimensiones de df_feats: {df_feats.shape}")

# 3. Análisis de clases
print("\n=== ANÁLISIS DE CLASES ===")
class_counts = df_classes['class'].value_counts()
print("Distribución de clases:")
print(class_counts)

# Visualización de la distribución de clases
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Distribución de Clases en el Dataset')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.savefig(os.path.join(results_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Análisis de características
print("\n=== ANÁLISIS DE CARACTERÍSTICAS ===")
# Estadísticas descriptivas
print("\nEstadísticas descriptivas de las características:")
print(df_feats.describe())

# Correlación entre características
print("\nCalculando correlaciones entre características...")
# Seleccionar solo las columnas numéricas para la correlación
numeric_cols = df_feats.select_dtypes(include=[np.number]).columns
correlation_matrix = df_feats[numeric_cols].corr()

# Visualizar correlaciones
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Matriz de Correlación entre Características')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Análisis de la red de transacciones
print("\n=== ANÁLISIS DE LA RED DE TRANSACCIONES ===")
# Construir el grafo
G = nx.from_pandas_edgelist(df_edgelist, 
                            source=df_edgelist.columns[0], 
                            target=df_edgelist.columns[1],
                            create_using=nx.DiGraph())

print(f"Número de nodos: {G.number_of_nodes()}")
print(f"Número de aristas: {G.number_of_edges()}")

# Calcular métricas de red
print("\nMétricas de red:")
print(f"Densidad: {nx.density(G):.6f}")
print(f"Diámetro: {nx.diameter(G) if nx.is_strongly_connected(G) else 'No conectado'}")
print(f"Promedio de grado: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# Distribución de grados
degrees = [d for n, d in G.degree()]
plt.figure(figsize=(10, 6))
sns.histplot(degrees, bins=50, kde=True)
plt.title('Distribución de Grados en la Red')
plt.xlabel('Grado')
plt.ylabel('Frecuencia')
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(results_dir, 'degree_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Análisis de componentes principales (PCA)
print("\n=== ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) ===")
# Preparar datos para PCA
feature_cols = [col for col in df_feats.columns if col != 'txid']
X = df_feats[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Visualizar varianza explicada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('Varianza Acumulada Explicada por Componentes Principales')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada Explicada')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'pca_variance.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualizar los dos primeros componentes principales
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('Visualización de los Dos Primeros Componentes Principales')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(os.path.join(results_dir, 'pca_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Análisis de valores atípicos
print("\n=== ANÁLISIS DE VALORES ATÍPICOS ===")
# Calcular el IQR para cada característica
Q1 = df_feats[feature_cols].quantile(0.25)
Q3 = df_feats[feature_cols].quantile(0.75)
IQR = Q3 - Q1

# Identificar valores atípicos
outliers = ((df_feats[feature_cols] < (Q1 - 1.5 * IQR)) | 
            (df_feats[feature_cols] > (Q3 + 1.5 * IQR))).sum()

print("\nNúmero de valores atípicos por característica:")
print(outliers)

# Visualizar valores atípicos para algunas características
plt.figure(figsize=(15, 10))
for i, col in enumerate(feature_cols[:9]):  # Mostrar solo las primeras 9 características
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df_feats[col])
    plt.title(f'Característica {col}')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'outliers_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8. Análisis de valores faltantes
print("\n=== ANÁLISIS DE VALORES FALTANTES ===")
missing_values = df_feats.isnull().sum()
print("\nNúmero de valores faltantes por característica:")
print(missing_values[missing_values > 0])

# 9. Guardar resumen del análisis
print("\n=== GUARDANDO RESUMEN DEL ANÁLISIS ===")
with open(os.path.join(results_dir, 'eda_summary.txt'), 'w') as f:
    f.write("=== RESUMEN DEL ANÁLISIS EXPLORATORIO DE DATOS ===\n\n")
    f.write(f"Dimensiones de df_classes: {df_classes.shape}\n")
    f.write(f"Dimensiones de df_edgelist: {df_edgelist.shape}\n")
    f.write(f"Dimensiones de df_feats: {df_feats.shape}\n\n")
    
    f.write("=== DISTRIBUCIÓN DE CLASES ===\n")
    f.write(str(class_counts) + "\n\n")
    
    f.write("=== MÉTRICAS DE RED ===\n")
    f.write(f"Número de nodos: {G.number_of_nodes()}\n")
    f.write(f"Número de aristas: {G.number_of_edges()}\n")
    f.write(f"Densidad: {nx.density(G):.6f}\n")
    f.write(f"Diámetro: {nx.diameter(G) if nx.is_strongly_connected(G) else 'No conectado'}\n")
    f.write(f"Promedio de grado: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}\n\n")
    
    f.write("=== ANÁLISIS DE VALORES ATÍPICOS ===\n")
    f.write("Número de valores atípicos por característica:\n")
    f.write(str(outliers) + "\n\n")
    
    f.write("=== ANÁLISIS DE VALORES FALTANTES ===\n")
    f.write("Número de valores faltantes por característica:\n")
    f.write(str(missing_values[missing_values > 0]) + "\n")

print(f"\nAnálisis exploratorio completado. Resultados guardados en {results_dir}") 