import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_class_distribution(df_classes, results_dir=None):
    """
    Visualiza la distribución de clases en el dataset.
    
    Args:
        df_classes: DataFrame con las clases de las transacciones
        results_dir: Directorio donde guardar la visualización
    """
    # Si no se proporciona un directorio, usar la ruta por defecto
    if results_dir is None:
        # Obtener la ruta absoluta al directorio raíz del proyecto
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        results_dir = os.path.join(root_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
    
    # Calcular distribución de clases
    class_counts = df_classes['class'].value_counts()
    
    # Visualizar distribución de clases
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribución de Clases en el Dataset')
    plt.xlabel('Clase')
    plt.ylabel('Frecuencia')
    plt.savefig(os.path.join(results_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_network_metrics(df_edgelist, results_dir=None):
    """
    Visualiza métricas de la red de transacciones.
    
    Args:
        df_edgelist: DataFrame con las aristas del grafo
        results_dir: Directorio donde guardar la visualización
    """
    # Si no se proporciona un directorio, usar la ruta por defecto
    if results_dir is None:
        # Obtener la ruta absoluta al directorio raíz del proyecto
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        results_dir = os.path.join(root_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
    
    # Construir el grafo
    df_edgelist.columns = df_edgelist.columns.str.lower().str.strip()
    df_edgelist.rename(columns={df_edgelist.columns[0]:'txid1',
                               df_edgelist.columns[1]:'txid2'},
                     inplace=True)
    G = nx.from_pandas_edgelist(df_edgelist,
                                source='txid1',
                                target='txid2',
                                create_using=nx.DiGraph())
    
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
    
    # Visualizar una muestra de la red (primeros 100 nodos)
    if G.number_of_nodes() > 100:
        sample_nodes = list(G.nodes())[:100]
        G_sample = G.subgraph(sample_nodes)
    else:
        G_sample = G
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_sample, seed=42)
    nx.draw(G_sample, pos, node_size=50, node_color='lightblue', 
            with_labels=False, arrows=True, edge_color='gray', alpha=0.7)
    plt.title('Muestra de la Red de Transacciones')
    plt.savefig(os.path.join(results_dir, 'network_sample.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_cols, results_dir=None):
    """
    Visualiza la importancia de las características según el modelo.
    
    Args:
        model: Modelo entrenado con atributo feature_importances_
        feature_cols: Lista de nombres de características
        results_dir: Directorio donde guardar la visualización
    """
    # Si no se proporciona un directorio, usar la ruta por defecto
    if results_dir is None:
        # Obtener la ruta absoluta al directorio raíz del proyecto
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        results_dir = os.path.join(root_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
    
    # Obtener importancia de características
    importances = model.feature_importances_
    
    # Crear DataFrame con importancia de características
    feat_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Visualizar las 20 características más importantes
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feat_importance.head(20))
    plt.title('Las 20 Características Más Importantes')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_distribution(df_full, results_dir=None):
    """
    Visualiza la distribución de las predicciones.
    
    Args:
        df_full: DataFrame con las predicciones
        results_dir: Directorio donde guardar la visualización
    """
    # Si no se proporciona un directorio, usar la ruta por defecto
    if results_dir is None:
        # Obtener la ruta absoluta al directorio raíz del proyecto
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        results_dir = os.path.join(root_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
    
    # Verificar si hay predicciones
    if 'predicted_semisup' not in df_full.columns:
        print("No hay predicciones para visualizar.")
        return
    
    # Calcular distribución de predicciones
    pred_counts = df_full['predicted_semisup'].value_counts()
    
    # Visualizar distribución de predicciones
    plt.figure(figsize=(10, 6))
    sns.barplot(x=pred_counts.index, y=pred_counts.values)
    plt.title('Distribución de Predicciones')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Frecuencia')
    plt.savefig(os.path.join(results_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualizar distribución de predicciones por clase original
    if 'class' in df_full.columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(x='class', hue='predicted_semisup', data=df_full)
        plt.title('Distribución de Predicciones por Clase Original')
        plt.xlabel('Clase Original')
        plt.ylabel('Frecuencia')
        plt.legend(title='Clase Predicha')
        plt.savefig(os.path.join(results_dir, 'prediction_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_confusion_matrix(y_true, y_pred, results_dir=None):
    """
    Visualiza la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        results_dir: Directorio donde guardar la visualización
    """
    # Si no se proporciona un directorio, usar la ruta por defecto
    if results_dir is None:
        # Obtener la ruta absoluta al directorio raíz del proyecto
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        results_dir = os.path.join(root_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
    
    # Calcular matriz de confusión
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Verdadera')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Este código se ejecutará solo si se ejecuta este script directamente
    pass 