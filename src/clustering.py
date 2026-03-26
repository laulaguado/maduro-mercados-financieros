#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para modelos de clustering de activos financieros.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import os
import logging
from tabulate import tabulate

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)


def preparar_features_clustering(df_ar, df_retornos, event_date):
    """
    Prepara las features para clustering de activos financieros.
    
    Args:
        df_ar (pandas.DataFrame): DataFrame con retornos anormales.
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        event_date (str or datetime): Fecha del evento.
    
    Returns:
        pandas.DataFrame: DataFrame con features para clustering.
    
    Example:
        >>> df_clustering = preparar_features_clustering(df_ar, df_retornos, '2026-01-03')
        >>> print(df_clustering.shape)
    """
    print("\n" + "="*80)
    print("PREPARACIÓN DE FEATURES PARA CLUSTERING")
    print("="*80)
    
    # Convertir event_date a datetime si es string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Crear columna de días relativos al evento
    df_ar_temp = df_ar.copy()
    df_ar_temp['dias_al_evento'] = (df_ar_temp.index - event_date).days
    
    df_ret_temp = df_retornos.copy()
    df_ret_temp['dias_al_evento'] = (df_ret_temp.index - event_date).days
    
    # Lista para almacenar datos por activo
    datos_activos = []
    
    # Obtener lista de activos (excluyendo SP500 que es el mercado)
    activos = [col.replace('AR_', '') for col in df_ar.columns if col.startswith('AR_')]
    
    for activo in activos:
        col_ar = f'AR_{activo}'
        
        # 1. CAR post-evento [0, +5]
        df_post5 = df_ar_temp[(df_ar_temp['dias_al_evento'] >= 0) & 
                              (df_ar_temp['dias_al_evento'] <= 5)]
        car_post5 = df_post5[col_ar].sum()
        
        # 2. Volatilidad promedio 20 días post-evento
        df_vol_post = df_ret_temp[(df_ret_temp['dias_al_evento'] >= 0) & 
                                  (df_ret_temp['dias_al_evento'] <= 20)]
        vol_post20 = df_vol_post[activo].std()
        
        # 3. CAR pre-evento [-5, -1]
        df_pre5 = df_ar_temp[(df_ar_temp['dias_al_evento'] >= -5) & 
                             (df_ar_temp['dias_al_evento'] <= -1)]
        car_pre5 = df_pre5[col_ar].sum()
        
        # 4. Correlación con Brent en ventana [-30, 0]
        df_corr_pre = df_ret_temp[(df_ret_temp['dias_al_evento'] >= -30) & 
                                  (df_ret_temp['dias_al_evento'] <= 0)]
        corr_brent = df_corr_pre[activo].corr(df_corr_pre['BRENT'])
        
        # 5. Cambio en correlación Brent pre vs post evento
        df_corr_post = df_ret_temp[(df_ret_temp['dias_al_evento'] >= 0) & 
                                   (df_ret_temp['dias_al_evento'] <= 30)]
        corr_brent_post = df_corr_post[activo].corr(df_corr_post['BRENT'])
        delta_corr = corr_brent_post - corr_brent
        
        # 6. Sector codificado
        from feature_engineering import calcular_sector
        sector = calcular_sector(activo)
        
        # Almacenar datos
        datos_activos.append({
            'activo': activo,
            'car_post5': car_post5,
            'vol_post20': vol_post20,
            'car_pre5': car_pre5,
            'corr_brent': corr_brent,
            'delta_corr': delta_corr,
            'sector': sector
        })
    
    # Crear DataFrame
    df_clustering = pd.DataFrame(datos_activos)
    df_clustering = df_clustering.set_index('activo')
    
    # Codificar sector
    le = LabelEncoder()
    df_clustering['sector_encoded'] = le.fit_transform(df_clustering['sector'])
    
    # Eliminar columna de sector original
    df_clustering = df_clustering.drop('sector', axis=1)
    
    # Aplicar Z-score
    scaler = StandardScaler()
    df_clustering_std = pd.DataFrame(
        scaler.fit_transform(df_clustering),
        index=df_clustering.index,
        columns=df_clustering.columns
    )
    
    # Aplicar winsorización (1%-99%)
    for col in df_clustering_std.columns:
        limite_inf = df_clustering_std[col].quantile(0.01)
        limite_sup = df_clustering_std[col].quantile(0.99)
        df_clustering_std[col] = df_clustering_std[col].clip(lower=limite_inf, upper=limite_sup)
    
    # Imprimir resumen
    print("\nFeatures calculadas para clustering:")
    print(f"- Número de activos: {len(df_clustering_std)}")
    print(f"- Features: {', '.join(df_clustering_std.columns)}")
    
    print("\nEstadísticas de las features (después de estandarización y winsorización):")
    print(df_clustering_std.describe().round(4))
    
    return df_clustering_std


def aplicar_kmeans(df_clustering, k_min=2, k_max=6):
    """
    Aplica K-Means para diferentes valores de k y selecciona el óptimo.
    
    Args:
        df_clustering (pandas.DataFrame): DataFrame con features para clustering.
        k_min (int): Valor mínimo de k.
        k_max (int): Valor máximo de k.
    
    Returns:
        dict: Diccionario con resultados de cada k.
    
    Example:
        >>> resultados_kmeans = aplicar_kmeans(df_clustering, k_min=2, k_max=6)
        >>> print(resultados_kmeans['k_optimo'])
    """
    print("\n" + "="*80)
    print(f"APLICACIÓN DE K-MEANS (k={k_min} a {k_max})")
    print("="*80)
    
    # Crear directorio para gráficos si no existe
    directorio = os.path.join("data", "processed", "graficos")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Diccionario para almacenar resultados
    resultados = {
        'k': [],
        'inercia': [],
        'silhouette': [],
        'modelos': {}
    }
    
    # Aplicar K-Means para cada k
    for k in range(k_min, k_max + 1):
        print(f"\nAplicando K-Means con k={k}...")
        
        # Entrenar modelo
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(df_clustering)
        
        # Calcular métricas
        inercia = kmeans.inertia_
        silhouette = silhouette_score(df_clustering, labels)
        
        # Almacenar resultados
        resultados['k'].append(k)
        resultados['inercia'].append(inercia)
        resultados['silhouette'].append(silhouette)
        resultados['modelos'][k] = kmeans
        
        print(f"  Inercia: {inercia:.4f}")
        print(f"  Silhouette Score: {silhouette:.4f}")
    
    # Gráfico del método del codo
    plt.figure(figsize=(10, 6))
    plt.plot(resultados['k'], resultados['inercia'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inercia', fontsize=12)
    plt.title('Método del Codo para Selección de k', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    ruta_codo = os.path.join(directorio, "metodo_codo.png")
    plt.savefig(ruta_codo, dpi=300, bbox_inches='tight')
    print(f"\nGráfico del método del codo guardado en: {ruta_codo}")
    
    # Gráfico de Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.plot(resultados['k'], resultados['silhouette'], 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs Número de Clusters', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    ruta_silhouette = os.path.join(directorio, "silhouette_vs_k.png")
    plt.savefig(ruta_silhouette, dpi=300, bbox_inches='tight')
    print(f"Gráfico de Silhouette Score guardado en: {ruta_silhouette}")
    
    # Seleccionar k óptimo (mayor Silhouette Score)
    idx_optimo = np.argmax(resultados['silhouette'])
    k_optimo = resultados['k'][idx_optimo]
    silhouette_optimo = resultados['silhouette'][idx_optimo]
    
    resultados['k_optimo'] = k_optimo
    resultados['silhouette_optimo'] = silhouette_optimo
    
    print(f"\nK óptimo recomendado: {k_optimo} (Silhouette Score = {silhouette_optimo:.4f})")
    
    return resultados


def aplicar_clustering_jerarquico(df_clustering):
    """
    Aplica clustering jerárquico y genera dendrograma.
    
    Args:
        df_clustering (pandas.DataFrame): DataFrame con features para clustering.
    
    Returns:
        sklearn.cluster.AgglomerativeClustering: Modelo ajustado.
    
    Example:
        >>> modelo_jerarquico = aplicar_clustering_jerarquico(df_clustering)
        >>> print(modelo_jerarquico.labels_)
    """
    print("\n" + "="*80)
    print("APLICACIÓN DE CLUSTERING JERÁRQUICO")
    print("="*80)
    
    # Crear directorio para gráficos si no existe
    directorio = os.path.join("data", "processed", "graficos")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Calcular linkage
    Z = linkage(df_clustering, method='ward')
    
    # Generar dendrograma
    plt.figure(figsize=(12, 8))
    dendrogram(
        Z,
        labels=df_clustering.index.tolist(),
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.xlabel('Activos', fontsize=12)
    plt.ylabel('Distancia', fontsize=12)
    plt.title('Dendrograma de Clustering Jerárquico', fontsize=14)
    plt.tight_layout()
    
    # Guardar dendrograma
    ruta_dendrograma = os.path.join(directorio, "dendrograma.png")
    plt.savefig(ruta_dendrograma, dpi=300, bbox_inches='tight')
    print(f"Dendrograma guardado en: {ruta_dendrograma}")
    
    # Determinar número óptimo de clusters por mayor salto en distancia
    distancias = Z[:, 2]
    saltos = np.diff(distancias)
    idx_mayor_salto = np.argmax(saltos)
    
    # El número de clusters es n - idx_mayor_salto
    n_clusters_sugerido = len(df_clustering) - idx_mayor_salto
    
    print(f"\nNúmero de clusters sugerido por mayor salto en distancia: {n_clusters_sugerido}")
    
    # Entrenar modelo con número sugerido de clusters
    modelo = AgglomerativeClustering(n_clusters=n_clusters_sugerido, linkage='ward')
    labels = modelo.fit_predict(df_clustering)
    
    # Imprimir distribución de clusters
    print("\nDistribución de activos por cluster:")
    for cluster in range(n_clusters_sugerido):
        activos_cluster = df_clustering.index[labels == cluster].tolist()
        print(f"Cluster {cluster}: {', '.join(activos_cluster)}")
    
    return modelo


def aplicar_dbscan(df_clustering, eps_range, min_samples_range):
    """
    Aplica DBSCAN con búsqueda de hiperparámetros.
    
    Args:
        df_clustering (pandas.DataFrame): DataFrame con features para clustering.
        eps_range (list): Rango de valores para eps.
        min_samples_range (list): Rango de valores para min_samples.
    
    Returns:
        sklearn.cluster.DBSCAN: Mejor modelo DBSCAN.
    
    Example:
        >>> dbscan = aplicar_dbscan(df_clustering, [0.5, 1.0, 1.5], [2, 3, 4])
        >>> print(dbscan.labels_)
    """
    print("\n" + "="*80)
    print("APLICACIÓN DE DBSCAN")
    print("="*80)
    
    mejor_silhouette = -1
    mejor_eps = None
    mejor_min_samples = None
    mejor_modelo = None
    
    # Probar combinaciones
    for eps in eps_range:
        for min_samples in min_samples_range:
            # Entrenar modelo
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df_clustering)
            
            # Verificar si hay más de 1 cluster (excluyendo ruido)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                # Calcular Silhouette Score (excluyendo ruido)
                mascara = labels != -1
                if np.sum(mascara) > 1:
                    silhouette = silhouette_score(df_clustering[mascara], labels[mascara])
                    
                    print(f"\nDBSCAN con eps={eps}, min_samples={min_samples}:")
                    print(f"  Clusters: {n_clusters}")
                    print(f"  Puntos de ruido: {np.sum(labels == -1)}")
                    print(f"  Silhouette Score: {silhouette:.4f}")
                    
                    # Actualizar mejor modelo
                    if silhouette > mejor_silhouette:
                        mejor_silhouette = silhouette
                        mejor_eps = eps
                        mejor_min_samples = min_samples
                        mejor_modelo = dbscan
    
    if mejor_modelo is None:
        print("\nNo se encontró una configuración válida de DBSCAN")
        print("Usando configuración por defecto: eps=1.0, min_samples=2")
        mejor_modelo = DBSCAN(eps=1.0, min_samples=2)
        mejor_modelo.fit(df_clustering)
        mejor_eps = 1.0
        mejor_min_samples = 2
    
    # Imprimir resultados del mejor modelo
    labels = mejor_modelo.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = np.sum(labels == -1)
    
    print(f"\nMejor configuración DBSCAN:")
    print(f"- eps: {mejor_eps}")
    print(f"- min_samples: {mejor_min_samples}")
    print(f"- Clusters: {n_clusters}")
    print(f"- Puntos de ruido: {n_ruido}")
    
    if n_ruido > 0:
        activos_ruido = df_clustering.index[labels == -1].tolist()
        print(f"- Activos marcados como ruido: {', '.join(activos_ruido)}")
    
    return mejor_modelo


def comparar_clusterings(kmeans_labels, jerarquico_labels, dbscan_labels, df_clustering):
    """
    Compara los tres métodos de clustering.
    
    Args:
        kmeans_labels (array): Etiquetas de K-Means.
        jerarquico_labels (array): Etiquetas de clustering jerárquico.
        dbscan_labels (array): Etiquetas de DBSCAN.
        df_clustering (pandas.DataFrame): DataFrame con features.
    
    Returns:
        array: Mejor conjunto de etiquetas.
    
    Example:
        >>> mejores_labels = comparar_clusterings(kmeans_labels, jerarquico_labels, dbscan_labels, df_clustering)
        >>> print(mejores_labels)
    """
    print("\n" + "="*80)
    print("COMPARACIÓN DE MÉTODOS DE CLUSTERING")
    print("="*80)
    
    # Calcular métricas para cada método
    metricas = {}
    
    # K-Means
    n_clusters_kmeans = len(set(kmeans_labels))
    if n_clusters_kmeans > 1:
        silhouette_kmeans = silhouette_score(df_clustering, kmeans_labels)
        davies_kmeans = davies_bouldin_score(df_clustering, kmeans_labels)
        metricas['K-Means'] = {
            'n_clusters': n_clusters_kmeans,
            'silhouette': silhouette_kmeans,
            'davies_bouldin': davies_kmeans
        }
    
    # Jerárquico
    n_clusters_jerarquico = len(set(jerarquico_labels))
    if n_clusters_jerarquico > 1:
        silhouette_jerarquico = silhouette_score(df_clustering, jerarquico_labels)
        davies_jerarquico = davies_bouldin_score(df_clustering, jerarquico_labels)
        metricas['Jerárquico'] = {
            'n_clusters': n_clusters_jerarquico,
            'silhouette': silhouette_jerarquico,
            'davies_bouldin': davies_jerarquico
        }
    
    # DBSCAN
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    if n_clusters_dbscan > 1:
        mascara = dbscan_labels != -1
        silhouette_dbscan = silhouette_score(df_clustering[mascara], dbscan_labels[mascara])
        davies_dbscan = davies_bouldin_score(df_clustering[mascara], dbscan_labels[mascara])
        metricas['DBSCAN'] = {
            'n_clusters': n_clusters_dbscan,
            'silhouette': silhouette_dbscan,
            'davies_bouldin': davies_dbscan
        }
    
    # Crear tabla comparativa
    tabla = []
    for metodo, met in metricas.items():
        tabla.append([
            metodo,
            met['n_clusters'],
            f"{met['silhouette']:.4f}",
            f"{met['davies_bouldin']:.4f}"
        ])
    
    headers = ["Método", "Clusters", "Silhouette", "Davies-Bouldin"]
    print(tabulate(tabla, headers=headers, tablefmt="grid"))
    
    # Seleccionar el mejor método (mayor Silhouette, menor Davies-Bouldin)
    mejor_metodo = max(metricas.items(), 
                       key=lambda x: (x[1]['silhouette'], -x[1]['davies_bouldin']))[0]
    
    print(f"\nMétodo recomendado: {mejor_metodo}")
    
    # Retornar etiquetas del mejor método
    if mejor_metodo == 'K-Means':
        return kmeans_labels
    elif mejor_metodo == 'Jerárquico':
        return jerarquico_labels
    else:
        return dbscan_labels


def graficar_clusters_pca(df_clustering, labels, nombres_activos, titulo='Clustering de Activos Financieros'):
    """
    Genera gráfico de clusters usando PCA para reducción de dimensionalidad.
    
    Args:
        df_clustering (pandas.DataFrame): DataFrame con features.
        labels (array): Etiquetas de cluster.
        nombres_activos (list): Nombres de los activos.
        titulo (str): Título del gráfico.
    
    Returns:
        matplotlib.figure.Figure: Figura con el gráfico.
    
    Example:
        >>> fig = graficar_clusters_pca(df_clustering, labels, df_clustering.index, 'Mi Clustering')
        >>> plt.show()
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO DE CLUSTERS CON PCA")
    print("="*80)
    
    # Crear directorio para gráficos si no existe
    directorio = os.path.join("data", "processed", "graficos")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(df_clustering)
    
    # Crear DataFrame para graficar
    df_pca = pd.DataFrame({
        'PC1': componentes[:, 0],
        'PC2': componentes[:, 1],
        'cluster': labels,
        'activo': nombres_activos
    })
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar puntos por cluster
    clusters_unicos = sorted(df_pca['cluster'].unique())
    colores = plt.cm.tab10(np.linspace(0, 1, len(clusters_unicos)))
    
    for i, cluster in enumerate(clusters_unicos):
        mascara = df_pca['cluster'] == cluster
        plt.scatter(
            df_pca.loc[mascara, 'PC1'],
            df_pca.loc[mascara, 'PC2'],
            c=[colores[i]],
            label=f'Cluster {cluster}',
            s=100,
            alpha=0.7
        )
    
    # Etiquetar cada punto con el nombre del activo
    for idx, fila in df_pca.iterrows():
        plt.annotate(
            fila['activo'],
            (fila['PC1'], fila['PC2']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
        )
    
    # Configurar gráfico
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)', fontsize=12)
    plt.title(titulo, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    ruta_guardado = os.path.join(directorio, "clustering_activos.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {ruta_guardado}")
    
    return plt.gcf()


def interpretar_clusters(df_clustering, labels, nombres_activos):
    """
    Genera interpretación financiera de cada cluster.
    
    Args:
        df_clustering (pandas.DataFrame): DataFrame con features.
        labels (array): Etiquetas de cluster.
        nombres_activos (list): Nombres de los activos.
    
    Returns:
        dict: Diccionario con interpretaciones por cluster.
    
    Example:
        >>> interpretaciones = interpretar_clusters(df_clustering, labels, df_clustering.index)
        >>> print(interpretaciones[0])
    """
    print("\n" + "="*80)
    print("INTERPRETACIÓN DE CLUSTERS")
    print("="*80)
    
    # Crear DataFrame con etiquetas
    df_interpretacion = df_clustering.copy()
    df_interpretacion['cluster'] = labels
    df_interpretacion['activo'] = nombres_activos
    
    # Diccionario para almacenar interpretaciones
    interpretaciones = {}
    
    # Para cada cluster
    for cluster in sorted(set(labels)):
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster}")
        print(f"{'='*60}")
        
        # Filtrar activos del cluster
        df_cluster = df_interpretacion[df_interpretacion['cluster'] == cluster]
        activos = df_cluster['activo'].tolist()
        
        # Calcular promedios de features
        promedios = df_cluster.drop(['cluster', 'activo'], axis=1).mean()
        
        # Imprimir activos
        print(f"\nActivos en el cluster: {', '.join(activos)}")
        
        # Imprimir promedios de features
        print("\nPromedio de features:")
        for feature, valor in promedios.items():
            print(f"- {feature}: {valor:.4f}")
        
        # Generar interpretación automática
        car_post = promedios['car_post5']
        vol_post = promedios['vol_post20']
        car_pre = promedios['car_pre5']
        corr_brent = promedios['corr_brent']
        delta_corr = promedios['delta_corr']
        
        # Determinar tipo de cluster
        if car_post > 0.5:
            tipo = "Activos reactivos positivos"
            descripcion = f"reaccionaron positivamente de forma inmediata (CAR post-evento: {car_post*100:.1f}%)"
        elif car_post < -0.5:
            tipo = "Activos reactivos negativos"
            descripcion = f"reaccionaron negativamente de forma inmediata (CAR post-evento: {car_post*100:.1f}%)"
        elif abs(car_post) <= 0.5 and vol_post > 1.0:
            tipo = "Activos de alta volatilidad"
            descripcion = f"mostraron alta volatilidad post-evento ({vol_post*100:.1f}%) sin dirección clara"
        elif abs(delta_corr) > 0.3:
            tipo = "Activos con cambio de correlación"
            descripcion = f"cambiaron significativamente su correlación con Brent ({delta_corr:+.2f})"
        else:
            tipo = "Activos estables"
            descripcion = "mantuvieron un comportamiento estable durante el evento"
        
        # Construir interpretación completa
        interpretacion = (
            f"Cluster {cluster} — {tipo}: {', '.join(activos)}. "
            f"CAR promedio post-evento: {car_post*100:+.1f}%. "
            f"Volatilidad promedio: {vol_post*100:.1f}%. "
            f"Correlación con Brent: {corr_brent:.2f}. "
            f"Cambio en correlación: {delta_corr:+.2f}. "
            f"Estos activos {descripcion}."
        )
        
        print(f"\nInterpretación: {interpretacion}")
        
        # Almacenar interpretación
        interpretaciones[cluster] = {
            'activos': activos,
            'promedios': promedios.to_dict(),
            'tipo': tipo,
            'interpretacion': interpretacion
        }
    
    return interpretaciones


if __name__ == "__main__":
    # Ejemplo de uso del módulo
    import os
    from data_collection import EVENT_DATE
    
    try:
        # Cargar datos
        ruta_ar = os.path.join("data", "processed", "retornos_anormales.csv")
        df_ar = pd.read_csv(ruta_ar, index_col=0, parse_dates=True)
        
        ruta_retornos = os.path.join("data", "processed", "retornos_diarios.csv")
        df_retornos = pd.read_csv(ruta_retornos, index_col=0, parse_dates=True)
        
        # Preparar features
        df_clustering = preparar_features_clustering(df_ar, df_retornos, EVENT_DATE)
        
        # Aplicar K-Means
        resultados_kmeans = aplicar_kmeans(df_clustering)
        k_optimo = resultados_kmeans['k_optimo']
        kmeans_labels = resultados_kmeans['modelos'][k_optimo].labels_
        
        # Aplicar clustering jerárquico
        modelo_jerarquico = aplicar_clustering_jerarquico(df_clustering)
        jerarquico_labels = modelo_jerarquico.labels_
        
        # Aplicar DBSCAN
        dbscan = aplicar_dbscan(df_clustering, [0.5, 1.0, 1.5, 2.0], [2, 3, 4])
        dbscan_labels = dbscan.labels_
        
        # Comparar métodos
        mejores_labels = comparar_clusterings(kmeans_labels, jerarquico_labels, dbscan_labels, df_clustering)
        
        # Graficar clusters
        fig = graficar_clusters_pca(df_clustering, mejores_labels, df_clustering.index)
        
        # Interpretar clusters
        interpretaciones = interpretar_clusters(df_clustering, mejores_labels, df_clustering.index)
        
        print("\nClustering completado con éxito.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ejecute primero los módulos anteriores para generar los datos necesarios.")