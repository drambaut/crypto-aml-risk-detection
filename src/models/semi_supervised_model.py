import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import os

class SemiSupervisedModel:
    """
    Clase para implementar un modelo semi-supervisado para la detección de fraudes.
    """
    
    def __init__(self, n_neighbors=10, max_iter=1000, n_estimators=100, random_state=42):
        """
        Inicializa el modelo semi-supervisado.
        
        Args:
            n_neighbors: Número de vecinos para Label Propagation
            max_iter: Número máximo de iteraciones para Label Propagation
            n_estimators: Número de árboles para Random Forest
            random_state: Semilla aleatoria para reproducibilidad
        """
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Inicializar modelos
        self.lp = LabelPropagation(kernel='knn', 
                                  n_neighbors=n_neighbors, 
                                  max_iter=max_iter)
        self.clf = RandomForestClassifier(n_estimators=n_estimators, 
                                         random_state=random_state, 
                                         n_jobs=-1)
        
        # Almacenar resultados
        self.y_pred = None
        self.y_proba = None
        self.classification_report = None
        self.roc_auc = None
    
    def fit(self, X, y, mask_known):
        """
        Entrena el modelo semi-supervisado.
        
        Args:
            X: Matriz de características
            y: Vector de etiquetas
            mask_known: Máscara para las transacciones con etiquetas conocidas
        """
        # 1. Label Propagation para etiquetas pseudo-supervisadas
        self.lp.fit(X, y)
        
        # 2. Preparar datos para entrenamiento supervisado
        X_known = X[mask_known]
        y_known = self.lp.transduction_[mask_known]
        
        # 3. Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_known, y_known,
            test_size=0.2, stratify=y_known, random_state=self.random_state
        )
        
        # 4. Entrenar Random Forest
        self.clf.fit(X_train, y_train)
        
        # 5. Evaluar en conjunto de prueba
        self.y_pred = self.clf.predict(X_test)
        idx1 = list(self.clf.classes_).index(1)
        self.y_proba = self.clf.predict_proba(X_test)[:, idx1]
        
        # 6. Calcular métricas
        self.classification_report = classification_report(y_test, self.y_pred)
        self.roc_auc = roc_auc_score(y_test, self.y_proba)
        
        return self
    
    def predict(self, X, mask_unknown):
        """
        Realiza predicciones para transacciones desconocidas.
        
        Args:
            X: Matriz de características
            mask_unknown: Máscara para las transacciones con etiquetas desconocidas
        
        Returns:
            Vector de predicciones para transacciones desconocidas
        """
        return self.clf.predict(X[mask_unknown])
    
    def predict_proba(self, X, mask_unknown):
        """
        Realiza predicciones de probabilidad para transacciones desconocidas.
        
        Args:
            X: Matriz de características
            mask_unknown: Máscara para las transacciones con etiquetas desconocidas
        
        Returns:
            Matriz de probabilidades para transacciones desconocidas
        """
        return self.clf.predict_proba(X[mask_unknown])
    
    def get_results(self):
        """
        Obtiene los resultados del modelo.
        
        Returns:
            Diccionario con los resultados del modelo
        """
        return {
            'classification_report': self.classification_report,
            'roc_auc': self.roc_auc
        }
    
    def save_results(self, df_full, mask_unknown, results_dir=None):
        """
        Guarda los resultados del modelo.
        
        Args:
            df_full: DataFrame completo con todas las transacciones
            mask_unknown: Máscara para las transacciones con etiquetas desconocidas
            results_dir: Directorio donde guardar los resultados
        
        Returns:
            DataFrame con las predicciones agregadas
        """
        # Obtener solo las columnas numéricas para predicción
        feature_cols = [col for col in df_full.columns if col != 'txid']
        X_features = df_full[feature_cols].values
        
        # Realizar predicciones
        df_full.loc[mask_unknown, 'predicted_semisup'] = self.predict(X_features, mask_unknown)
        
        # Guardar métricas si se proporciona un directorio
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            
            # Guardar métricas
            with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
                f.write("=== MÉTRICAS DEL MODELO SEMI-SUPERVISADO ===\n\n")
                f.write("Reporte de Clasificación:\n")
                f.write(self.classification_report)
                f.write(f"\nROC AUC: {self.roc_auc:.4f}")
        
        return df_full

if __name__ == "__main__":
    # Este código se ejecutará solo si se ejecuta este script directamente
    pass 