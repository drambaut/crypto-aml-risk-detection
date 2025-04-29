#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para la detección de fraudes en transacciones de Bitcoin.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Importar módulos del proyecto
from data.loader import load_data, prepare_data_for_training
from features.graph_features import extract_all_features
from models.semi_supervised_model import SemiSupervisedModel
from visualization.visualizer import (
    plot_class_distribution, 
    plot_network_metrics, 
    plot_feature_importance,
    plot_prediction_distribution
)

def run_exploratory_analysis(df_classes, df_edgelist, df_feats, results_dir):
    """
    Realiza el análisis exploratorio de los datos.
    
    Args:
        df_classes: DataFrame con las clases
        df_edgelist: DataFrame con la lista de aristas
        df_feats: DataFrame con las características
        results_dir: Directorio donde guardar los resultados
    """
    print("\n=== ANÁLISIS EXPLORATORIO DE DATOS ===")
    
    # 1. Análisis básico de los datos
    print("\nDimensiones de los DataFrames:")
    print(f"df_classes: {df_classes.shape}")
    print(f"df_edgelist: {df_edgelist.shape}")
    print(f"df_feats: {df_feats.shape}")
    
    # 2. Análisis de clases
    print("\nDistribución de clases:")
    class_counts = df_classes['class'].value_counts()
    print(class_counts)
    
    # Visualización de la distribución de clases
    plot_class_distribution(df_classes, results_dir)
    
    # 3. Análisis de la red
    print("\nAnálisis de la red de transacciones:")
    G = nx.from_pandas_edgelist(df_edgelist, 
                               source=df_edgelist.columns[0], 
                               target=df_edgelist.columns[1],
                               create_using=nx.DiGraph())
    
    print(f"Número de nodos: {G.number_of_nodes()}")
    print(f"Número de aristas: {G.number_of_edges()}")
    print(f"Densidad de la red: {nx.density(G):.4f}")
    
    # Visualización de métricas de red
    plot_network_metrics(df_edgelist, results_dir)
    
    # 4. Análisis de características
    print("\nEstadísticas descriptivas de las características:")
    print(df_feats.describe())
    
    # 5. Análisis de correlaciones
    print("\nCalculando correlaciones entre características...")
    numeric_cols = df_feats.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_feats[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title('Matriz de Correlación entre Características')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 6. Análisis de componentes principales
    print("\nRealizando análisis de componentes principales...")
    feature_cols = [col for col in df_feats.columns if col != 'txid']
    X = df_feats[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title('Varianza Acumulada Explicada por Componentes Principales')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Acumulada Explicada')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'pca_variance.png'))
    plt.close()
    
    print("\nAnálisis exploratorio completado.")

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Detección de fraudes en transacciones de Bitcoin')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Ruta al archivo de configuración YAML')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directorio donde se encuentran los archivos de datos')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directorio donde guardar los resultados')
    parser.add_argument('--n_components', type=int, default=None,
                        help='Número de componentes para SVD')
    parser.add_argument('--n_neighbors', type=int, default=None,
                        help='Número de vecinos para Label Propagation')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Número máximo de iteraciones para Label Propagation')
    parser.add_argument('--n_estimators', type=int, default=None,
                        help='Número de árboles para Random Forest')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Semilla aleatoria para reproducibilidad')
    parser.add_argument('--skip_eda', action='store_true',
                        help='Saltar el análisis exploratorio de datos')
    parser.add_argument('--run_eda', action='store_true',
                        help='Ejecutar el análisis exploratorio de datos')
    return parser.parse_args()

def train_final_model(X, y, feature_cols, results_dir, models_dir, config):
    """
    Entrena un modelo final usando todas las etiquetas disponibles.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas
        feature_cols: Lista de nombres de características
        results_dir: Directorio para guardar resultados
        models_dir: Directorio para guardar modelos
        config: Configuración del modelo
    """
    print("\nEntrenando modelo final con todas las etiquetas...")
    
    # Obtener parámetros de configuración
    n_estimators = config['model']['final']['n_estimators']
    random_state = config['model']['final']['random_state']
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Entrenar modelo final
    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=config['execution']['n_jobs']
    )
    final_model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    from sklearn.metrics import classification_report, roc_auc_score
    print("\n=== MÉTRICAS DEL MODELO FINAL ===")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Visualizar importancia de características
    plot_feature_importance(final_model, feature_cols, results_dir, prefix='final_model')
    
    # Guardar modelo final
    final_model_path = os.path.join(models_dir, 'final_model.joblib')
    joblib.dump(final_model, final_model_path)
    print(f"Modelo final guardado en: {final_model_path}")
    
    return final_model

def load_config(args):
    """
    Carga la configuración desde el archivo YAML y la combina con los argumentos de línea de comandos.
    
    Args:
        args: Argumentos de línea de comandos
    
    Returns:
        Diccionario con la configuración combinada
    """
    # Cargar configuración desde archivo YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Sobrescribir con argumentos de línea de comandos si se proporcionan
    if args.data_dir:
        config['directories']['data'] = args.data_dir
    if args.results_dir:
        config['directories']['results'] = args.results_dir
    if args.n_components:
        config['features']['n_components'] = args.n_components
    if args.n_neighbors:
        config['model']['semi_supervised']['n_neighbors'] = args.n_neighbors
    if args.max_iter:
        config['model']['semi_supervised']['max_iter'] = args.max_iter
    if args.n_estimators:
        config['model']['semi_supervised']['n_estimators'] = args.n_estimators
        config['model']['final']['n_estimators'] = args.n_estimators
    if args.random_state:
        config['model']['semi_supervised']['random_state'] = args.random_state
        config['model']['final']['random_state'] = args.random_state
    
    # Configurar EDA
    if args.skip_eda:
        config['execution']['run_eda'] = False
    if args.run_eda:
        config['execution']['run_eda'] = True
    
    return config

def main():
    """Función principal."""
    # Parsear argumentos
    args = parse_args()
    
    # Cargar configuración
    config = load_config(args)
    
    # Obtener la ruta absoluta al directorio raíz del proyecto
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Establecer directorios
    data_dir = os.path.join(root_dir, config['directories']['data'])
    results_dir = os.path.join(root_dir, config['directories']['results'])
    models_dir = os.path.join(root_dir, config['directories']['models'])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Carga de datos
    print("Cargando datos...")
    df_classes, df_edgelist, df_feats = load_data(data_dir)
    
    # 2. Análisis exploratorio de datos
    if config['execution']['run_eda']:
        run_exploratory_analysis(df_classes, df_edgelist, df_feats, results_dir)
    
    # 3. Extracción de características
    print("\nExtrayendo características de grafo...")
    df_full = extract_all_features(df_edgelist, df_feats, n_components=config['features']['n_components'])
    
    # 4. Preparación de datos para entrenamiento
    print("Preparando datos para entrenamiento...")
    X, y, feature_cols, mask_known = prepare_data_for_training(df_classes, df_feats, df_full)
    
    # 5. Entrenamiento del modelo semi-supervisado
    print("Entrenando modelo semi-supervisado...")
    model = SemiSupervisedModel(
        n_neighbors=config['model']['semi_supervised']['n_neighbors'],
        max_iter=config['model']['semi_supervised']['max_iter'],
        n_estimators=config['model']['semi_supervised']['n_estimators'],
        random_state=config['model']['semi_supervised']['random_state']
    )
    model.fit(X, y, mask_known)
    
    # 6. Guardar el modelo semi-supervisado
    print("Guardando modelo semi-supervisado...")
    model_path = os.path.join(models_dir, 'semi_supervised_model.joblib')
    joblib.dump(model, model_path)
    print(f"Modelo semi-supervisado guardado en: {model_path}")
    
    # 7. Obtener predicciones para datos no etiquetados
    print("Obteniendo predicciones para datos no etiquetados...")
    mask_unknown = ~mask_known
    y_pred_unknown = model.predict(X, mask_unknown)
    
    # 8. Crear conjunto de datos completo con todas las etiquetas
    y_complete = y.copy()
    y_complete[mask_unknown] = y_pred_unknown
    
    # 9. Entrenar modelo final con todas las etiquetas
    final_model = train_final_model(
        X, y_complete, feature_cols, results_dir, models_dir, config
    )
    
    # 10. Visualización de resultados
    print("Generando visualizaciones de resultados...")
    plot_feature_importance(model.clf, feature_cols, results_dir, prefix='semi_supervised')
    
    # 11. Guardar resultados
    print("Guardando resultados...")
    df_full = model.save_results(df_full, mask_unknown, results_dir)
    
    # 12. Visualización de predicciones
    print("Generando visualizaciones de predicciones...")
    plot_prediction_distribution(df_full, results_dir)
    
    # 13. Imprimir métricas del modelo semi-supervisado
    results = model.get_results()
    print("\n=== MÉTRICAS DEL MODELO SEMI-SUPERVISADO ===")
    print("Reporte de Clasificación:")
    print(results['classification_report'])
    print(f"ROC AUC: {results['roc_auc']}")
    
    print(f"\nProceso completado. Resultados guardados en {results_dir}")

if __name__ == "__main__":
    main()