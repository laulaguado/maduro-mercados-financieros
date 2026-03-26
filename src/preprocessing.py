#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la limpieza y transformación de datos financieros.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)


def imputar_nulos_forward_fill(df):
    """
    Aplica forward fill (ffill) para imputar valores faltantes en series financieras.
    
    En series financieras diarias, el precio del día festivo es igual al último 
    precio disponible, por lo que forward fill es una técnica adecuada para 
    imputar valores faltantes.
    
    Args:
        df (pandas.DataFrame): DataFrame con posibles valores nulos.
    
    Returns:
        pandas.DataFrame: DataFrame con valores imputados.
    
    Example:
        >>> df_imputado = imputar_nulos_forward_fill(df_precios)
        >>> print(df_imputado.isnull().sum())
    """
    print("\n" + "="*80)
    print("IMPUTACIÓN DE VALORES NULOS CON FORWARD FILL")
    print("="*80)
    
    # Contar nulos antes de imputación
    nulos_antes = df.isnull().sum()
    porcentaje_antes = (nulos_antes / len(df)) * 100
    
    # Imputar con forward fill
    df_imputado = df.ffill()
    
    # Si quedan nulos al inicio, usar backward fill
    df_imputado = df_imputado.bfill()
    
    # Contar nulos después de imputación
    nulos_despues = df_imputado.isnull().sum()
    porcentaje_despues = (nulos_despues / len(df_imputado)) * 100
    
    # Imprimir resumen
    print("\nNULOS ANTES DE IMPUTACIÓN:")
    for col in df.columns:
        if nulos_antes[col] > 0:
            print(f"  - {col}: {nulos_antes[col]} nulos ({porcentaje_antes[col]:.2f}%)")
    
    print("\nNULOS DESPUÉS DE IMPUTACIÓN:")
    for col in df_imputado.columns:
        if nulos_despues[col] > 0:
            print(f"  - {col}: {nulos_despues[col]} nulos ({porcentaje_despues[col]:.2f}%)")
        else:
            print(f"  - {col}: 0 nulos (0.00%)")
    
    return df_imputado


def detectar_outliers_iqr(df_retornos, umbral=3.0):
    """
    Detecta outliers usando el método IQR × umbral para cada columna.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos financieros.
        umbral (float): Multiplicador del IQR para considerar un valor como outlier.
    
    Returns:
        pandas.DataFrame: DataFrame original con columnas adicionales de marcado de outliers.
    
    Example:
        >>> df_con_outliers = detectar_outliers_iqr(df_retornos, umbral=3.0)
        >>> print(df_con_outliers.filter(like='_es_outlier').sum())
    """
    print("\n" + "="*80)
    print(f"DETECCIÓN DE OUTLIERS (MÉTODO IQR × {umbral})")
    print("="*80)
    
    df_resultado = df_retornos.copy()
    conteo_outliers = {}
    
    for columna in df_retornos.columns:
        # Calcular Q1, Q3 e IQR
        Q1 = df_retornos[columna].quantile(0.25)
        Q3 = df_retornos[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites
        limite_inferior = Q1 - umbral * IQR
        limite_superior = Q3 + umbral * IQR
        
        # Crear columna de marcado
        nombre_columna_outlier = f"{columna}_es_outlier"
        df_resultado[nombre_columna_outlier] = ((df_retornos[columna] < limite_inferior) | 
                                               (df_retornos[columna] > limite_superior))
        
        # Contar outliers
        conteo_outliers[columna] = df_resultado[nombre_columna_outlier].sum()
    
    # Imprimir resumen
    print("\nRESUMEN DE OUTLIERS DETECTADOS:")
    print(f"{'Activo':<10} | {'Outliers':<8} | {'Porcentaje':>10}")
    print("-" * 32)
    
    for columna, conteo in conteo_outliers.items():
        porcentaje = (conteo / len(df_retornos)) * 100
        print(f"{columna:<10} | {conteo:<8} | {porcentaje:>8.2f}%")
    
    print(f"\nTotal de outliers: {sum(conteo_outliers.values())}")
    
    return df_resultado


def winsorizacion(df_retornos, percentil_inf=0.01, percentil_sup=0.99):
    """
    Aplica winsorización a las columnas de retornos.
    
    Limita los valores extremos al percentil especificado, útil para reducir
    el impacto de outliers en análisis de clustering.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos financieros.
        percentil_inf (float): Percentil inferior para winsorización (0-1).
        percentil_sup (float): Percentil superior para winsorización (0-1).
    
    Returns:
        pandas.DataFrame: DataFrame con valores winsorizados.
    
    Example:
        >>> df_wins = winsorizacion(df_retornos, 0.01, 0.99)
        >>> print(df_wins.describe())
    """
    print("\n" + "="*80)
    print(f"WINSORIZACIÓN (PERCENTILES {percentil_inf*100}% - {percentil_sup*100}%)")
    print("="*80)
    
    df_wins = df_retornos.copy()
    
    # Filtrar solo columnas de retornos (excluir columnas _es_outlier)
    columnas_retornos = [col for col in df_retornos.columns if not col.endswith('_es_outlier')]
    
    for columna in columnas_retornos:
        # Calcular límites
        limite_inf = df_retornos[columna].quantile(percentil_inf)
        limite_sup = df_retornos[columna].quantile(percentil_sup)
        
        # Aplicar winsorización
        df_wins[columna] = df_retornos[columna].clip(lower=limite_inf, upper=limite_sup)
        
        # Calcular estadísticas antes y después
        media_antes = df_retornos[columna].mean()
        std_antes = df_retornos[columna].std()
        media_despues = df_wins[columna].mean()
        std_despues = df_wins[columna].std()
        
        print(f"\nActivo: {columna}")
        print(f"  - Límite inferior: {limite_inf:.6f}")
        print(f"  - Límite superior: {limite_sup:.6f}")
        print(f"  - Media antes: {media_antes:.6f} | después: {media_despues:.6f}")
        print(f"  - Desv. estándar antes: {std_antes:.6f} | después: {std_despues:.6f}")
    
    return df_wins


def estandarizar_zscore(df):
    """
    Aplica estandarización Z-score (media=0, std=1).
    
    Args:
        df (pandas.DataFrame): DataFrame a estandarizar.
    
    Returns:
        tuple: (DataFrame estandarizado, objeto StandardScaler entrenado)
    
    Example:
        >>> df_std, scaler = estandarizar_zscore(df_features)
        >>> print(df_std.describe().round(2))
    """
    print("\n" + "="*80)
    print("ESTANDARIZACIÓN Z-SCORE")
    print("="*80)
    
    # Crear y entrenar el scaler
    scaler = StandardScaler()
    df_std = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    # Imprimir estadísticas
    print("\nEstadísticas antes de estandarización:")
    print(df.describe().loc[['mean', 'std']].round(4))
    
    print("\nEstadísticas después de estandarización:")
    print(df_std.describe().loc[['mean', 'std']].round(4))
    
    return df_std, scaler


def generar_estadisticas_descriptivas(df_retornos):
    """
    Calcula estadísticas descriptivas completas para cada columna.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos financieros.
    
    Returns:
        pandas.DataFrame: DataFrame con estadísticas descriptivas.
    
    Example:
        >>> stats = generar_estadisticas_descriptivas(df_retornos)
        >>> print(stats)
    """
    print("\n" + "="*80)
    print("ESTADÍSTICAS DESCRIPTIVAS")
    print("="*80)
    
    # Calcular estadísticas
    stats = pd.DataFrame({
        'media': df_retornos.mean(),
        'mediana': df_retornos.median(),
        'desv_std': df_retornos.std(),
        'asimetria': df_retornos.skew(),
        'curtosis': df_retornos.kurtosis(),
        'p5': df_retornos.quantile(0.05),
        'p95': df_retornos.quantile(0.95),
        'min': df_retornos.min(),
        'max': df_retornos.max()
    }).T
    
    # Imprimir tabla formateada
    print("\nESTADÍSTICAS DESCRIPTIVAS DE RETORNOS:")
    print(stats.round(4))
    
    # Identificar distribuciones con fat tails
    activos_fat_tails = stats.loc['curtosis'][stats.loc['curtosis'] > 3].index.tolist()
    if activos_fat_tails:
        print("\nACTIVOS CON COLAS PESADAS (CURTOSIS > 3):")
        for activo in activos_fat_tails:
            print(f"  - {activo}: curtosis = {stats.loc['curtosis', activo]:.4f}")
    
    # Identificar distribuciones con asimetría significativa
    activos_asimetricos = stats.loc['asimetria'][abs(stats.loc['asimetria']) > 0.5].index.tolist()
    if activos_asimetricos:
        print("\nACTIVOS CON ASIMETRÍA SIGNIFICATIVA (|ASIMETRÍA| > 0.5):")
        for activo in activos_asimetricos:
            asimetria = stats.loc['asimetria', activo]
            direccion = "negativa" if asimetria < 0 else "positiva"
            print(f"  - {activo}: asimetría {direccion} = {asimetria:.4f}")
    
    return stats


if __name__ == "__main__":
    # Ejemplo de uso del módulo
    import os
    from data_collection import calcular_retornos_logaritmicos
    
    # Cargar datos de ejemplo
    try:
        ruta_datos = os.path.join("data", "processed", "retornos_diarios.csv")
        df_retornos = pd.read_csv(ruta_datos, index_col=0, parse_dates=True)
        
        # Generar estadísticas descriptivas
        stats = generar_estadisticas_descriptivas(df_retornos)
        
        # Detectar outliers
        df_con_outliers = detectar_outliers_iqr(df_retornos)
        
        # Imputar nulos si existen
        df_imputado = imputar_nulos_forward_fill(df_retornos)
        
        # Winsorización
        df_wins = winsorizacion(df_retornos)
        
        # Estandarización
        df_std, scaler = estandarizar_zscore(df_retornos)
        
        print("\nProcesamiento completado con éxito.")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_datos}")
        print("Ejecute primero data_collection.py para generar los datos necesarios.")