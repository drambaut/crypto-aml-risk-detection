import unittest
import pandas as pd
import numpy as np
import os
import sys

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.semi_supervised_model import SemiSupervisedModel

class TestSemiSupervisedModel(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear datos de prueba más grandes y balanceados
        np.random.seed(42)
        n_samples = 20  # Aumentamos el número de muestras
        
        # Crear características
        self.X = np.random.randn(n_samples, 2)
        
        # Crear etiquetas balanceadas (10 de cada clase)
        self.y = np.array([0] * 10 + [1] * 10)
        
        # Marcar algunas etiquetas como conocidas, asegurando que haya ejemplos de ambas clases
        # 5 ejemplos de clase 0 y 5 ejemplos de clase 1 serán conocidos
        self.mask_known = np.array([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5)
        
        # Crear modelo
        self.model = SemiSupervisedModel(
            n_neighbors=2,
            max_iter=100,
            n_estimators=10,
            random_state=42
        )
    
    def test_fit(self):
        """Prueba el entrenamiento del modelo."""
        self.model.fit(self.X, self.y, self.mask_known)
        
        # Verificar que el modelo se entrenó correctamente
        self.assertTrue(hasattr(self.model, 'lp'))
        self.assertTrue(hasattr(self.model, 'clf'))
        
        # Verificar que las predicciones tienen la forma correcta
        self.assertTrue(hasattr(self.model, 'y_pred'))
        self.assertTrue(hasattr(self.model, 'y_proba'))
    
    def test_predict(self):
        """Prueba las predicciones del modelo."""
        # Entrenar modelo
        self.model.fit(self.X, self.y, self.mask_known)
        
        # Obtener predicciones
        mask_unknown = ~self.mask_known
        y_pred = self.model.predict(self.X, mask_unknown)
        
        # Verificar que las predicciones tienen la forma correcta
        self.assertEqual(len(y_pred), np.sum(mask_unknown))
        self.assertTrue(all(y_pred >= 0))
        self.assertTrue(all(y_pred <= 1))
    
    def test_predict_proba(self):
        """Prueba las probabilidades de predicción del modelo."""
        # Entrenar modelo
        self.model.fit(self.X, self.y, self.mask_known)
        
        # Obtener probabilidades
        mask_unknown = ~self.mask_known
        y_proba = self.model.predict_proba(self.X, mask_unknown)
        
        # Verificar que las probabilidades tienen la forma correcta
        self.assertEqual(y_proba.shape[0], np.sum(mask_unknown))
        self.assertEqual(y_proba.shape[1], 2)
        self.assertTrue(np.allclose(np.sum(y_proba, axis=1), 1.0))
    
    def test_get_results(self):
        """Prueba la obtención de resultados del modelo."""
        # Entrenar modelo
        self.model.fit(self.X, self.y, self.mask_known)
        
        # Obtener resultados
        results = self.model.get_results()
        
        # Verificar que los resultados tienen la forma correcta
        self.assertTrue('classification_report' in results)
        self.assertTrue('roc_auc' in results)
        self.assertTrue(isinstance(results['roc_auc'], float))
    
    def test_save_results(self):
        """Prueba el guardado de resultados del modelo."""
        # Crear directorio temporal para pruebas
        test_dir = os.path.join(os.path.dirname(__file__), 'test_results')
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # Entrenar modelo
            self.model.fit(self.X, self.y, self.mask_known)
            
            # Crear DataFrame de prueba
            df_full = pd.DataFrame({
                'txid': [f'tx{i}' for i in range(len(self.X))],
                'feat_1': self.X[:, 0],
                'feat_2': self.X[:, 1]
            })
            
            # Guardar resultados
            mask_unknown = ~self.mask_known
            df_full = self.model.save_results(df_full, mask_unknown, test_dir)
            
            # Verificar que se crearon los archivos
            self.assertTrue(os.path.exists(os.path.join(test_dir, 'model_metrics.txt')))
            self.assertTrue('predicted_semisup' in df_full.columns)
        
        finally:
            # Limpiar
            import shutil
            shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main() 