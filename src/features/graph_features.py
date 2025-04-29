import pandas as pd
import networkx as nx
import numpy as np
import os

def extract_graph_features(df_edgelist, df_feats):
    """
    Extrae características de grafo a partir de la lista de aristas.
    
    Args:
        df_edgelist: DataFrame con las aristas del grafo
        df_feats: DataFrame con las características de las transacciones
        
    Returns:
        DataFrame con las características de grafo
    """
    # Construir el grafo dirigido
    df_edgelist.columns = df_edgelist.columns.str.lower().str.strip()
    df_edgelist.rename(columns={df_edgelist.columns[0]:'txid1',
                               df_edgelist.columns[1]:'txid2'},
                     inplace=True)
    G = nx.from_pandas_edgelist(df_edgelist,
                                source='txid1',
                                target='txid2',
                                create_using=nx.DiGraph())
    
    # Calcular métricas de grafo
    in_deg = pd.Series(dict(G.in_degree()), name='in_degree')
    out_deg = pd.Series(dict(G.out_degree()), name='out_degree')
    pagerank = pd.Series(nx.pagerank(G), name='pagerank')
    
    # Calcular betweenness centrality con un subconjunto de nodos para eficiencia
    k = min(100, G.number_of_nodes())
    betw = pd.Series(nx.betweenness_centrality(G, k=k, normalized=True, seed=42),
                     name='betweenness')
    
    # Crear DataFrame con las características de grafo
    df_graph = (pd.concat([in_deg, out_deg, pagerank, betw], axis=1)
                .reset_index()
                .rename(columns={'index':'txid'}))
    
    return df_graph

def extract_embeddings(df_edgelist, df_feats, n_components=64):
    """
    Extrae embeddings de nodos usando Truncated SVD sobre la matriz de adyacencia.
    
    Args:
        df_edgelist: DataFrame con las aristas del grafo
        df_feats: DataFrame con las características de las transacciones
        n_components: Número de componentes para SVD
        
    Returns:
        DataFrame con los embeddings
    """
    # Construir el grafo
    df_edgelist.columns = df_edgelist.columns.str.lower().str.strip()
    df_edgelist.rename(columns={df_edgelist.columns[0]:'txid1',
                               df_edgelist.columns[1]:'txid2'},
                     inplace=True)
    G = nx.from_pandas_edgelist(df_edgelist,
                                source='txid1',
                                target='txid2',
                                create_using=nx.DiGraph())
    
    # Obtener lista de nodos
    nodes = list(G.nodes())
    
    # Crear matriz de adyacencia
    A = nx.adjacency_matrix(G, nodelist=nodes)
    
    # Aplicar Truncated SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(A)
    
    # Crear DataFrame con los embeddings
    emb_df = (pd.DataFrame(Z, index=nodes)
              .add_prefix('emb_')
              .reset_index()
              .rename(columns={'index':'txid'}))
    
    return emb_df

def extract_all_features(df_edgelist, df_feats, n_components=64):
    """
    Extrae todas las características de grafo y embeddings.
    
    Args:
        df_edgelist: DataFrame con las aristas del grafo
        df_feats: DataFrame con las características de las transacciones
        n_components: Número de componentes para SVD
        
    Returns:
        DataFrame con todas las características
    """
    # Extraer características de grafo
    df_graph = extract_graph_features(df_edgelist, df_feats)
    
    # Extraer embeddings
    emb_df = extract_embeddings(df_edgelist, df_feats, n_components)
    
    # Unir todas las características
    df_full = (df_feats
        .merge(df_graph, on='txid', how='left')
        .merge(emb_df, on='txid', how='left')
        .fillna(0))
    
    return df_full

if __name__ == "__main__":
    # Este código se ejecutará solo si se ejecuta este script directamente
    pass 