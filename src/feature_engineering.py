#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la creación de variables (feature engineering) para el análisis financiero.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
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


def calcular_volatilidad_historica(df_retornos, ventana=20):
    """
    Calcula la desviación estándar rodante de los últimos `ventana` días para cada activo.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        ventana (int): Tamaño de la ventana en días.
    
    Returns:
        pandas.DataFrame: DataFrame con volatilidades históricas.
    
    Example:
        >>> df_vol = calcular_volatilidad_historica(df_retornos, ventana=20)
        >>> print(df_vol.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO VOLATILIDAD HISTÓRICA (VENTANA {ventana} DÍAS)")
    print("="*80)
    
    df_vol = pd.DataFrame(index=df_retornos.index)
    
    for columna in df_retornos.columns:
        nombre_vol = f"{columna}_vol{ventana}"
        df_vol[nombre_vol] = df_retornos[columna].rolling(window=ventana).std()
    
    # Imprimir resumen
    print(f"\nVolatilidad histórica calculada con ventana de {ventana} días")
    print(f"- Shape: {df_vol.shape}")
    print(f"- Primeros valores disponibles desde: {df_vol.dropna().index[0]}")
    print(f"- Valores nulos: {df_vol.isnull().sum().sum()} (primeros {ventana-1} días)")
    
    return df_vol


def calcular_momentum(df_retornos, ventana=5):
    """
    Calcula el retorno acumulado de los últimos `ventana` días.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        ventana (int): Tamaño de la ventana en días.
    
    Returns:
        pandas.DataFrame: DataFrame con momentum.
    
    Example:
        >>> df_mom = calcular_momentum(df_retornos, ventana=5)
        >>> print(df_mom.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO MOMENTUM (VENTANA {ventana} DÍAS)")
    print("="*80)
    
    df_mom = pd.DataFrame(index=df_retornos.index)
    
    for columna in df_retornos.columns:
        nombre_mom = f"{columna}_mom{ventana}"
        # Suma de retornos logarítmicos = retorno acumulado
        df_mom[nombre_mom] = df_retornos[columna].rolling(window=ventana).sum()
    
    # Imprimir resumen
    print(f"\nMomentum calculado con ventana de {ventana} días")
    print(f"- Shape: {df_mom.shape}")
    print(f"- Primeros valores disponibles desde: {df_mom.dropna().index[0]}")
    print(f"- Valores nulos: {df_mom.isnull().sum().sum()} (primeros {ventana-1} días)")
    
    return df_mom


def calcular_correlacion_rodante_brent(df_retornos, ventana=30):
    """
    Calcula la correlación de Pearson de cada activo con BRENT en ventana rodante.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        ventana (int): Tamaño de la ventana en días.
    
    Returns:
        pandas.DataFrame: DataFrame con correlaciones rodantes.
    
    Example:
        >>> df_corr = calcular_correlacion_rodante_brent(df_retornos, ventana=30)
        >>> print(df_corr.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO CORRELACIÓN RODANTE CON BRENT (VENTANA {ventana} DÍAS)")
    print("="*80)
    
    # Verificar que BRENT esté en las columnas
    if 'BRENT' not in df_retornos.columns:
        raise ValueError("La columna 'BRENT' no existe en el DataFrame")
    
    df_corr = pd.DataFrame(index=df_retornos.index)
    
    for columna in df_retornos.columns:
        if columna != 'BRENT':  # No calcular correlación de BRENT consigo mismo
            nombre_corr = f"{columna}_corr_brent"
            df_corr[nombre_corr] = df_retornos[columna].rolling(window=ventana).corr(df_retornos['BRENT'])
    
    # Imprimir resumen
    print(f"\nCorrelación rodante con BRENT calculada con ventana de {ventana} días")
    print(f"- Shape: {df_corr.shape}")
    print(f"- Primeros valores disponibles desde: {df_corr.dropna().index[0]}")
    print(f"- Valores nulos: {df_corr.isnull().sum().sum()} (primeros {ventana-1} días)")
    
    return df_corr


def calcular_delta_vix(df_retornos):
    """
    Calcula la variación diaria del VIX.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
    
    Returns:
        pandas.Series: Serie con variación diaria del VIX.
    
    Example:
        >>> delta_vix = calcular_delta_vix(df_retornos)
        >>> print(delta_vix.head())
    """
    print("\n" + "="*80)
    print("CALCULANDO DELTA VIX (VARIACIÓN DIARIA)")
    print("="*80)
    
    # Verificar que VIX esté en las columnas
    if 'VIX' not in df_retornos.columns:
        raise ValueError("La columna 'VIX' no existe en el DataFrame")
    
    # Calcular la variación diaria del VIX
    delta_vix = df_retornos['VIX'] - df_retornos['VIX'].shift(1)
    delta_vix.name = 'DELTA_VIX'
    
    # Imprimir resumen
    print("\nVariación diaria del VIX calculada")
    print(f"- Longitud: {len(delta_vix)}")
    print(f"- Valores nulos: {delta_vix.isnull().sum()} (primer día)")
    print(f"- Media: {delta_vix.mean():.6f}")
    print(f"- Desviación estándar: {delta_vix.std():.6f}")
    
    return delta_vix


def crear_indicador_ventana(df, event_date, pre=10, post=60):
    """
    Crea columnas indicadoras de la ventana del evento.
    
    Args:
        df (pandas.DataFrame): DataFrame con índice de fechas.
        event_date (str): Fecha del evento en formato 'YYYY-MM-DD'.
        pre (int): Días previos al evento para la ventana pre-evento.
        post (int): Días posteriores al evento para la ventana post-evento.
    
    Returns:
        pandas.DataFrame: DataFrame con columnas indicadoras añadidas.
    
    Example:
        >>> df_ventana = crear_indicador_ventana(df, "2026-01-03", pre=10, post=60)
        >>> print(df_ventana['ventana_evento'].value_counts())
    """
    print("\n" + "="*80)
    print(f"CREANDO INDICADOR DE VENTANA DE EVENTO (PRE={pre}, POST={post})")
    print("="*80)
    
    df_resultado = df.copy()
    
    # Convertir event_date a datetime si es string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Crear columna de días al evento
    df_resultado['dias_al_evento'] = (df_resultado.index - event_date).days
    
    # Crear columna de ventana del evento
    condiciones = [
        (df_resultado['dias_al_evento'] >= -pre) & (df_resultado['dias_al_evento'] < 0),
        (df_resultado['dias_al_evento'] >= 0) & (df_resultado['dias_al_evento'] <= 5),
        (df_resultado['dias_al_evento'] > 5) & (df_resultado['dias_al_evento'] <= post)
    ]
    
    valores = ['pre_evento', 'evento', 'post_evento']
    
    df_resultado['ventana_evento'] = 'fuera'
    df_resultado['ventana_evento'] = np.select(condiciones, valores, default='fuera')
    
    # Imprimir resumen
    print("\nDistribución de observaciones por ventana:")
    print(df_resultado['ventana_evento'].value_counts())
    
    print("\nRango de días por ventana:")
    for ventana in ['pre_evento', 'evento', 'post_evento', 'fuera']:
        if ventana in df_resultado['ventana_evento'].values:
            dias = df_resultado.loc[df_resultado['ventana_evento'] == ventana, 'dias_al_evento']
            print(f"- {ventana}: {dias.min()} a {dias.max()} días")
    
    return df_resultado


def calcular_sector(nombre_activo):
    """
    Retorna el sector del activo como string.
    
    Args:
        nombre_activo (str): Nombre del activo financiero.
    
    Returns:
        str: Sector del activo.
    
    Example:
        >>> sector = calcular_sector('BRENT')
        >>> print(sector)
        'energia'
    """
    sectores = {
        'energia': ['BRENT', 'WTI', 'EXXON', 'CHEVRON'],
        'indice': ['SP500', 'COLCAP', 'BOVESPA'],
        'divisa': ['USD_COP'],
        'metal': ['GOLD', 'COPPER'],
        'volatilidad': ['VIX']
    }
    
    for sector, activos in sectores.items():
        if nombre_activo in activos:
            return sector
    
    return 'otro'


def construir_dataset_modelamiento(df_retornos, df_features):
    """
    Integra retornos y features en un único DataFrame para modelamiento.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        df_features (pandas.DataFrame): DataFrame con features calculadas.
    
    Returns:
        pandas.DataFrame: DataFrame completo para modelamiento.
    
    Example:
        >>> df_modelo = construir_dataset_modelamiento(df_retornos, df_features)
        >>> print(df_modelo.shape)
    """
    print("\n" + "="*80)
    print("CONSTRUYENDO DATASET FINAL PARA MODELAMIENTO")
    print("="*80)
    
    # Combinar retornos y features
    df_combinado = pd.concat([df_retornos, df_features], axis=1)
    
    # Añadir columna de sector
    for columna in df_retornos.columns:
        df_combinado[f'{columna}_sector'] = calcular_sector(columna)
    
    # Eliminar filas con NaN
    filas_antes = len(df_combinado)
    df_combinado = df_combinado.dropna()
    filas_eliminadas = filas_antes - len(df_combinado)
    
    # Imprimir resumen
    print(f"\nDataset final construido:")
    print(f"- Shape original: {filas_antes} filas")
    print(f"- Filas eliminadas por NaN: {filas_eliminadas} ({filas_eliminadas/filas_antes*100:.2f}%)")
    print(f"- Shape final: {df_combinado.shape}")
    
    # Distribución de sectores
    print("\nDistribución de sectores:")
    sectores = [calcular_sector(col) for col in df_retornos.columns]
    for sector in set(sectores):
        activos = [col for col in df_retornos.columns if calcular_sector(col) == sector]
        print(f"- {sector}: {len(activos)} activos ({', '.join(activos)})")
    
    return df_combinado


if __name__ == "__main__":
    # Ejemplo de uso del módulo
    import os
    from data_collection import EVENT_DATE
    
    # Cargar datos de ejemplo
    try:
        ruta_datos = os.path.join("data", "processed", "retornos_diarios.csv")
        df_retornos = pd.read_csv(ruta_datos, index_col=0, parse_dates=True)
        
        # Calcular features
        df_vol = calcular_volatilidad_historica(df_retornos)
        df_mom = calcular_momentum(df_retornos)
        df_corr = calcular_correlacion_rodante_brent(df_retornos)
        delta_vix = calcular_delta_vix(df_retornos)
        
        # Combinar features
        df_features = pd.concat([df_vol, df_mom, df_corr, delta_vix], axis=1)
        
        # Crear indicador de ventana
        df_features = crear_indicador_ventana(df_features, EVENT_DATE)
        
        # Construir dataset final
        df_modelo = construir_dataset_modelamiento(df_retornos, df_features)
        
        # Guardar dataset
        ruta_guardado = os.path.join("data", "processed", "dataset_modelamiento.csv")
        df_modelo.to_csv(ruta_guardado)
        print(f"\nDataset guardado en: {ruta_guardado}")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_datos}")
        print("Ejecute primero data_collection.py para generar los datos necesarios.")