import pandas as pd
import os

def load_data(data_dir=None):
    """
    Carga los datos del dataset Elliptic.
    
    Args:
        data_dir: Directorio donde se encuentran los archivos de datos.
                 Si es None, se usa la ruta por defecto.
    
    Returns:
        df_classes: DataFrame con las clases de las transacciones
        df_edgelist: DataFrame con las aristas del grafo
        df_feats: DataFrame con las características de las transacciones
    """
    # Si no se proporciona un directorio, usar la ruta por defecto
    if data_dir is None:
        # Obtener la ruta absoluta al directorio raíz del proyecto
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_dir = os.path.join(root_dir, 'data', 'elliptic_bitcoin_dataset')
    
    # Cargar los datos
    df_classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))
    df_edgelist = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))
    df_feats = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None)
    
    # Asignar nombres a las columnas de características
    df_feats.columns = ['txid'] + [f"feat_{i}" for i in range(1, df_feats.shape[1])]
    
    return df_classes, df_edgelist, df_feats

def prepare_data_for_training(df_classes, df_feats, df_full):
    """
    Prepara los datos para el entrenamiento.
    
    Args:
        df_classes: DataFrame con las clases de las transacciones
        df_feats: DataFrame con las características originales
        df_full: DataFrame con todas las características (incluyendo las de grafo)
    
    Returns:
        X: Matriz de características
        y: Vector de etiquetas
        feature_cols: Lista de nombres de características
        mask_known: Máscara para las transacciones con etiquetas conocidas
    """
    # Unir las clases con las características
    df_full = df_full.merge(df_classes.rename(columns={'TxId':'txid','txId':'txid'}), 
                           on='txid', how='left')
    
    # Convertir class a string, luego mapear correctamente
    df_full['class'] = df_full['class'].astype(str)
    y = df_full['class'].map({'0':0, '1':1}).fillna(-1).astype(int).values
    
    # Seleccionar características para el modelo
    feature_cols = [c for c in df_full.columns if c not in ['txid','class']]
    X = df_full[feature_cols].values
    
    # Crear máscara para transacciones con etiquetas conocidas
    mask_known = df_full['class'].isin(['0','1'])
    
    return X, y, feature_cols, mask_known

if __name__ == "__main__":
    # Este código se ejecutará solo si se ejecuta este script directamente
    pass 