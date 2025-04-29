import unittest
import pandas as pd
import numpy as np
import os
import sys

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.graph_features import extract_graph_features, extract_embeddings, extract_all_features

class TestGraphFeatures(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear datos de prueba
        self.df_edgelist = pd.DataFrame({
            'txId1': ['tx1', 'tx2', 'tx3'],
            'txId2': ['tx2', 'tx3', 'tx1']
        })
        
        self.df_feats = pd.DataFrame({
            'txid': ['tx1', 'tx2', 'tx3'],
            'feat_1': [1.0, 2.0, 3.0],
            'feat_2': [4.0, 5.0, 6.0]
        })
    
    def test_extract_graph_features(self):
        """Prueba la función extract_graph_features."""
        df_graph = extract_graph_features(self.df_edgelist, self.df_feats)
        
        # Verificar que el DataFrame tiene la forma correcta
        self.assertEqual(len(df_graph), 3)
        
        # Verificar que las columnas son correctas
        self.assertTrue('txid' in df_graph.columns)
        self.assertTrue('in_degree' in df_graph.columns)
        self.assertTrue('out_degree' in df_graph.columns)
        self.assertTrue('pagerank' in df_graph.columns)
        self.assertTrue('betweenness' in df_graph.columns)
        
        # Verificar que los grados son correctos
        self.assertEqual(df_graph.loc[df_graph['txid'] == 'tx1', 'in_degree'].iloc[0], 1)
        self.assertEqual(df_graph.loc[df_graph['txid'] == 'tx1', 'out_degree'].iloc[0], 1)
    
    def test_extract_embeddings(self):
        """Prueba la función extract_embeddings."""
        n_components = 2
        emb_df = extract_embeddings(self.df_edgelist, self.df_feats, n_components)
        
        # Verificar que el DataFrame tiene la forma correcta
        self.assertEqual(len(emb_df), 3)
        
        # Verificar que las columnas son correctas
        self.assertTrue('txid' in emb_df.columns)
        self.assertTrue('emb_0' in emb_df.columns)
        self.assertTrue('emb_1' in emb_df.columns)
    
    def test_extract_all_features(self):
        """Prueba la función extract_all_features."""
        n_components = 2
        df_full = extract_all_features(self.df_edgelist, self.df_feats, n_components)
        
        # Verificar que el DataFrame tiene la forma correcta
        self.assertEqual(len(df_full), 3)
        
        # Verificar que las columnas son correctas
        expected_cols = ['txid', 'feat_1', 'feat_2', 'in_degree', 'out_degree', 
                        'pagerank', 'betweenness', 'emb_0', 'emb_1']
        self.assertTrue(all(col in df_full.columns for col in expected_cols))
        
        # Verificar que no hay valores nulos
        self.assertFalse(df_full.isnull().any().any())

if __name__ == '__main__':
    unittest.main() 