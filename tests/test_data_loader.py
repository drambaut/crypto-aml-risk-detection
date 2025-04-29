import unittest
import pandas as pd
import numpy as np
import os
import sys

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import load_data, prepare_data_for_training

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear datos de prueba
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Crear DataFrames de prueba
        self.df_classes = pd.DataFrame({
            'txId': ['tx1', 'tx2', 'tx3'],
            'class': ['0', '1', '0']
        })
        
        self.df_edgelist = pd.DataFrame({
            'txId1': ['tx1', 'tx2'],
            'txId2': ['tx2', 'tx3']
        })
        
        self.df_feats = pd.DataFrame({
            'txid': ['tx1', 'tx2', 'tx3'],
            'feat_1': [1.0, 2.0, 3.0],
            'feat_2': [4.0, 5.0, 6.0]
        })
        
        # Guardar datos de prueba
        self.df_classes.to_csv(os.path.join(self.test_data_dir, 'elliptic_txs_classes.csv'), index=False)
        self.df_edgelist.to_csv(os.path.join(self.test_data_dir, 'elliptic_txs_edgelist.csv'), index=False)
        self.df_feats.to_csv(os.path.join(self.test_data_dir, 'elliptic_txs_features.csv'), index=False, header=False)
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        import shutil
        shutil.rmtree(self.test_data_dir)
    
    def test_load_data(self):
        """Prueba la función load_data."""
        df_classes, df_edgelist, df_feats = load_data(self.test_data_dir)
        
        # Verificar que los DataFrames tienen la forma correcta
        self.assertEqual(len(df_classes), 3)
        self.assertEqual(len(df_edgelist), 2)
        self.assertEqual(len(df_feats), 3)
        
        # Verificar que las columnas son correctas
        self.assertTrue('txId' in df_classes.columns)
        self.assertTrue('class' in df_classes.columns)
        self.assertTrue('txId1' in df_edgelist.columns)
        self.assertTrue('txId2' in df_edgelist.columns)
        self.assertTrue('txid' in df_feats.columns)
    
    def test_prepare_data_for_training(self):
        """Prueba la función prepare_data_for_training."""
        # Crear DataFrame completo para pruebas
        df_full = pd.DataFrame({
            'txid': ['tx1', 'tx2', 'tx3'],
            'feat_1': [1.0, 2.0, 3.0],
            'feat_2': [4.0, 5.0, 6.0]
        })
        
        # Preparar datos
        X, y, feature_cols, mask_known = prepare_data_for_training(
            self.df_classes, self.df_feats, df_full
        )
        
        # Verificar dimensiones
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(len(feature_cols), 2)
        self.assertEqual(len(mask_known), 3)
        
        # Verificar que las etiquetas son correctas
        self.assertTrue(all(y[mask_known] >= 0))
        
        # Verificar que las características son correctas
        self.assertTrue(all(col in feature_cols for col in ['feat_1', 'feat_2']))

if __name__ == '__main__':
    unittest.main() 