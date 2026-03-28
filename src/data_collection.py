#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la descarga y configuración de datos financieros.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diccionario global de tickers
TICKERS = {
    'SP500': '^GSPC',
    'VIX': '^VIX',
    'BRENT': 'BZ=F',
    'WTI': 'CL=F',
    'COLCAP': '^COLCAP',
    'BOVESPA': '^BVSP',
    'IBVC': '^IBVC',  # Índice Bursátil de Caracas (Venezuela)
    'MERVAL': '^MERV',  # Índice Merval (Argentina)
    'USD_COP': 'USDCOP=X',
    'GOLD': 'GC=F',
    'COPPER': 'HG=F',
    'EXXON': 'XOM',
    'CHEVRON': 'CVX'
}

# Constantes globales
START_DATE = "2020-01-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
EVENT_DATE = "2026-01-03"
EVENT_DATE_HABIL = "2026-01-05"  # primer día hábil post-captura


def descargar_datos(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """
    Descarga precios de cierre ajustados desde Yahoo Finance.
    
    Args:
        tickers (dict): Diccionario con nombres amigables y símbolos de Yahoo Finance.
        start (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end (str): Fecha de fin en formato 'YYYY-MM-DD'.
    
    Returns:
        pandas.DataFrame: DataFrame con fechas como índice y precios de cierre ajustados.
    
    Example:
        >>> df_precios = descargar_datos(TICKERS, "2020-01-01", "2022-12-31")
        >>> print(df_precios.shape)
        (756, 11)
    """
    print("\n" + "="*80)
    print(f"DESCARGANDO DATOS FINANCIEROS: {start} hasta {end}")
    print("="*80)
    
    # Lista para almacenar DataFrames de cada activo
    dfs = []
    
    # Invertir el diccionario para mapear símbolos a nombres amigables
    simbolo_a_nombre = {v: k for k, v in tickers.items()}
    
    # Descargar datos para cada ticker
    for nombre, simbolo in tickers.items():
        try:
            print(f"Descargando {nombre} ({simbolo})...")
            data = yf.download(simbolo, start=start, end=end, progress=False)
            
            if data.empty:
                print(f"⚠️ No se encontraron datos para {nombre} ({simbolo})")
                continue
            
            # Intentar obtener precio de cierre ajustado
            if 'Adj Close' in data.columns:
                df_activo = data[['Adj Close']].copy()
            elif 'Close' in data.columns:
                df_activo = data[['Close']].copy()
            else:
                print(f"⚠️ No se encontró columna de cierre para {nombre} ({simbolo})")
                continue
            
            df_activo.columns = [nombre]
            
            dfs.append(df_activo)
            print(f"✓ {nombre}: {len(df_activo)} registros descargados")
            
        except Exception as e:
            print(f"❌ Error al descargar {nombre} ({simbolo}): {str(e)}")
    
    # Combinar todos los DataFrames
    if not dfs:
        raise ValueError("No se pudo descargar ningún activo financiero")
        
    df_combinado = pd.concat(dfs, axis=1)
    
    # Resumen de la descarga
    print("\nRESUMEN DE DESCARGA:")
    print(f"- Período: {start} a {end}")
    print(f"- Shape: {df_combinado.shape}")
    print(f"- Rango de fechas: {df_combinado.index.min()} a {df_combinado.index.max()}")
    print(f"- Activos descargados: {', '.join(df_combinado.columns)}")
    
    return df_combinado


def calcular_retornos_logaritmicos(df_precios):
    """
    Calcula retornos logarítmicos para cada columna del DataFrame.
    
    Args:
        df_precios (pandas.DataFrame): DataFrame con precios de cierre ajustados.
    
    Returns:
        pandas.DataFrame: DataFrame con retornos logarítmicos.
    
    Example:
        >>> df_retornos = calcular_retornos_logaritmicos(df_precios)
        >>> print(df_retornos.head())
    """
    print("\n" + "="*80)
    print("CALCULANDO RETORNOS LOGARÍTMICOS")
    print("="*80)
    
    # Calcular retornos logarítmicos: r_t = ln(P_t / P_{t-1})
    df_retornos = np.log(df_precios / df_precios.shift(1))
    
    # Eliminar primera fila (NaN por el shift)
    df_retornos = df_retornos.iloc[1:].copy()
    
    print(f"- Shape retornos: {df_retornos.shape}")
    print(f"- Rango de fechas: {df_retornos.index.min()} a {df_retornos.index.max()}")
    
    return df_retornos


def validar_calidad_datos(df):
    """
    Verifica la calidad de los datos financieros.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros.
    
    Returns:
        dict: Diccionario con métricas de calidad.
    
    Example:
        >>> metricas = validar_calidad_datos(df_precios)
        >>> print(metricas['porcentaje_nulos'])
    """
    print("\n" + "="*80)
    print("VALIDACIÓN DE CALIDAD DE DATOS")
    print("="*80)
    
    metricas = {}
    
    # Verificar si hay precios negativos
    negativos = (df < 0).sum()
    if negativos.sum() > 0:
        print("⚠️ ADVERTENCIA: Se encontraron precios negativos:")
        print(negativos[negativos > 0])
    else:
        print("✓ No se encontraron precios negativos")
    metricas['precios_negativos'] = negativos.to_dict()
    
    # Verificar fechas duplicadas
    duplicados = df.index.duplicated().sum()
    if duplicados > 0:
        print(f"⚠️ ADVERTENCIA: Se encontraron {duplicados} fechas duplicadas")
    else:
        print("✓ No se encontraron fechas duplicadas")
    metricas['fechas_duplicadas'] = duplicados
    
    # Verificar valores nulos
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df)) * 100
    
    if nulos.sum() > 0:
        print("⚠️ ADVERTENCIA: Se encontraron valores nulos:")
        for col in df.columns:
            if nulos[col] > 0:
                print(f"  - {col}: {nulos[col]} nulos ({porcentaje_nulos[col]:.2f}%)")
    else:
        print("✓ No se encontraron valores nulos")
    
    metricas['nulos'] = nulos.to_dict()
    metricas['porcentaje_nulos'] = porcentaje_nulos.to_dict()
    
    # Resumen estadístico básico
    print("\nRESUMEN ESTADÍSTICO:")
    print(f"- Número de registros: {len(df)}")
    print(f"- Número de columnas: {len(df.columns)}")
    print(f"- Rango de fechas: {df.index.min()} a {df.index.max()}")
    
    return metricas


def guardar_datos(df, nombre_archivo):
    """
    Guarda DataFrame en formato CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame a guardar.
        nombre_archivo (str): Nombre del archivo sin extensión.
    
    Returns:
        str: Ruta completa donde se guardó el archivo.
    
    Example:
        >>> ruta = guardar_datos(df_retornos, "retornos_diarios")
        >>> print(f"Datos guardados en: {ruta}")
    """
    # Crear directorio si no existe
    directorio = os.path.join("data", "processed")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Ruta completa del archivo
    ruta_completa = os.path.join(directorio, f"{nombre_archivo}.csv")
    
    # Guardar DataFrame
    df.to_csv(ruta_completa)
    
    print(f"\n✓ Datos guardados en: {ruta_completa}")
    print(f"  - Shape: {df.shape}")
    
    return ruta_completa


if __name__ == "__main__":
    # Ejemplo de uso del módulo
    print(f"Fecha de inicio: {START_DATE}")
    print(f"Fecha de fin: {END_DATE}")
    print(f"Fecha del evento: {EVENT_DATE}")
    
    # Descargar datos
    df_precios = descargar_datos()
    
    # Validar calidad
    metricas = validar_calidad_datos(df_precios)
    
    # Calcular retornos
    df_retornos = calcular_retornos_logaritmicos(df_precios)
    
    # Guardar datos
    guardar_datos(df_precios, "precios_diarios")
    guardar_datos(df_retornos, "retornos_diarios")