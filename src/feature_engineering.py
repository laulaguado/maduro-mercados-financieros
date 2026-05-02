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
import os

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


# =============================================================================
# NUEVAS FUNCIONES DE FEATURE ENGINEERING
# =============================================================================

def calcular_rsi(df, ventana=14):
    """
    Calcula el Índice de Fuerza Relativa (RSI) para cada activo.
    
    Args:
        df (pandas.DataFrame): DataFrame con precios o retornos.
        ventana (int): Tamaño de la ventana para el cálculo (default 14).
    
    Returns:
        pandas.DataFrame: DataFrame con el RSI para cada columna.
    
    Example:
        >>> df_rsi = calcular_rsi(df_precios, ventana=14)
        >>> print(df_rsi.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO RSI (VENTANA {ventana} DÍAS)")
    print("="*80)
    
    delta = df.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(ventana).mean()
    avg_loss = losses.rolling(ventana).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.columns = [f"{col}_rsi{ventana}" for col in rsi.columns]
    
    print(f"\nRSI calculado con ventana de {ventana} días")
    print(f"- Shape: {rsi.shape}")
    print(f"- Primeros valores disponibles desde: {rsi.dropna().index[0]}")
    print(f"- Valores nulos: {rsi.isnull().sum().sum()} (primeros {ventana} días)")
    print(f"- Rango: [{rsi.min().min():.2f}, {rsi.max().max():.2f}]")
    
    return rsi


def calcular_bandas_bollinger(df, ventana=20):
    """
    Calcula la distancia del precio al promedio móvil en desviaciones estándar (Bollinger Z-score).
    
    Args:
        df (pandas.DataFrame): DataFrame con precios o retornos.
        ventana (int): Tamaño de la ventana para el promedio móvil (default 20).
    
    Returns:
        pandas.DataFrame: DataFrame con el Z-score de Bollinger para cada columna.
    
    Example:
        >>> df_bb = calcular_bandas_bollinger(df_precios, ventana=20)
        >>> print(df_bb.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO BANDAS DE BOLLINGER (VENTANA {ventana} DÍAS)")
    print("="*80)
    
    media = df.rolling(ventana).mean()
    std = df.rolling(ventana).std()
    z = (df - media) / std
    z.columns = [f"{col}_bollinger{ventana}" for col in z.columns]
    
    print(f"\nBandas de Bollinger calculadas con ventana de {ventana} días")
    print(f"- Shape: {z.shape}")
    print(f"- Primeros valores disponibles desde: {z.dropna().index[0]}")
    print(f"- Valores nulos: {z.isnull().sum().sum()} (primeros {ventana-1} días)")
    print(f"- Rango Z-score: [{z.min().min():.2f}, {z.max().max():.2f}]")
    
    return z


def calcular_ratio_volumen(df_volumen, ventana=20):
    """
    Calcula el ratio del volumen del día versus el promedio de los últimos
    N días. Detecta actividad inusual ( spikes de volumen).

    Args:
        df_volumen (pandas.DataFrame): DataFrame con volúmenes de trading.
        ventana (int): Ventana para el promedio móvil (default 20).

    Returns:
        pandas.DataFrame: DataFrame con ratio volumen/promedio.

    Example:
        >>> df_ratio_vol = calcular_ratio_volumen(df_volumen, ventana=20)
        >>> print(df_ratio_vol.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO RATIO VOLUMEN/PROMEDIO ({ventana} DÍAS)")
    print("="*80)

    df_ratio = pd.DataFrame(index=df_volumen.index)

    for columna in df_volumen.columns:
        # Volumen promedio rodante
        volumen_promedio = df_volumen[columna].rolling(window=ventana).mean()

        # Ratio: volumen actual / promedio
        ratio = df_volumen[columna] / volumen_promedio

        df_ratio[f"{columna}_vol_ratio"] = ratio

    # Imprimir resumen
    print(f"\nRatio volumen calculado con ventana de {ventana} días")
    print(f"- Shape: {df_ratio.shape}")
    print(f"- Primeros valores disponibles desde: {df_ratio.dropna().index[0]}")
    print(f"- Valores nulos: {df_ratio.isnull().sum().sum()} (primeros {ventana-1} días)")
    print(f"- Rango de ratios: [{df_ratio.min().min():.2f}, {df_ratio.max().max():.2f}]")
    print(f"  (ratio > 1.5 indica volumen 50% superior al promedio)")

    return df_ratio


def calcular_retorno_mensual(df, ventana=21):
    """
    Calcula el retorno acumulado de los últimos N días hábiles (aproximadamente mensual).
    
    Args:
        df (pandas.DataFrame): DataFrame con retornos logarítmicos.
        ventana (int): Número de días hábiles para el retorno acumulado (default 21).
    
    Returns:
        pandas.DataFrame: DataFrame con el retorno acumulado.
    
    Example:
        >>> df_ret_mensual = calcular_retorno_mensual(df_retornos, ventana=21)
        >>> print(df_ret_mensual.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO RETORNO MENSUAL (VENTANA {ventana} DÍAS)")
    print("="*80)
    
    ret_mensual = df.rolling(ventana).sum()
    ret_mensual.columns = [f"{col}_ret{ventana}d" for col in ret_mensual.columns]
    
    print(f"\nRetorno mensual calculado con ventana de {ventana} días")
    print(f"- Shape: {ret_mensual.shape}")
    print(f"- Primeros valores disponibles desde: {ret_mensual.dropna().index[0]}")
    print(f"- Valores nulos: {ret_mensual.isnull().sum().sum()} (primeros {ventana-1} días)")
    
    return ret_mensual


def agregar_features_macro(df, fecha_inicio, fecha_fin):
    """
    Descarga indicadores macroeconómicos con yfinance y calcula sus retornos logarítmicos.
    
    Indicadores descargados:
      - DXY (US Dollar Index): 'DX-Y.NYB'
      - Gas natural (Natural Gas): 'NG=F'
      - Treasury 10 años: '^TNX'
      - Treasury 2 años: '^IRX'
    
    Calcula additionally el spread 10Y-2Y (diferencia de yields).
    
    Args:
        df (pandas.DataFrame): DataFrame de referencia para el índice de fechas.
        fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        fecha_fin (str): Fecha de fin en formato 'YYYY-MM-DD'.
    
    Returns:
        pandas.DataFrame: DataFrame con retornos logarítmicos de los indicadores macro
        y el spread 10Y-2Y. Columnas: DXY, GAS, T10Y, T2Y, spread_10y2y.
    
    Example:
        >>> df_macro = agregar_features_macro(df_retornos, '2020-01-01', '2026-04-30')
        >>> print(df_macro.head())
    """
    print("\n" + "="*80)
    print("DESCARGANDO FEATURES MACROECONÓMICAS CON YFINANCE")
    print("="*80)
    print(f"Periodo: {fecha_inicio} a {fecha_fin}")
    
    # Intentar importar yfinance
    try:
        import yfinance as yf
    except ImportError:
        print("\n⚠ ERROR: yfinance no está instalado.")
        print("   Instálelo con: pip install yfinance")
        return pd.DataFrame(index=df.index)
    
    # Símbolos a descargar
    simbolos = {
        'DXY': 'DX-Y.NYB',
        'GAS': 'NG=F',
        'T10Y': '^TNX',
        'T2Y': '^IRX'
    }
    
    datos = {}
    for nombre, simbolo in simbolos.items():
        try:
            print(f"  Descargando {nombre} ({simbolo})...")
            serie = yf.download(
                simbolo,
                start=fecha_inicio,
                end=fecha_fin,
                auto_adjust=True,
                progress=False
            )['Close']
            # Calcular retorno logarítmico diario
            datos[nombre] = np.log(serie / serie.shift(1))
            print(f"    ✓ {nombre}: {len(serie)} observaciones")
        except Exception as e:
            print(f"    ✗ Error descargando {nombre}: {e}")
            datos[nombre] = None
    
    # Crear DataFrame
    df_macro = pd.DataFrame(datos, index=df.index)
    
    # Calcular spread 10Y-2Y
    if 'T10Y' in df_macro.columns and 'T2Y' in df_macro.columns:
        df_macro['spread_10y2y'] = df_macro['T10Y'] - df_macro['T2Y']
        print(f"  ✓ spread_10y2y calculado")
    else:
        print(f"  ⚠ spread_10y2y no calculado (faltan T10Y o T2Y)")
    
    # Rellenar datos faltantes
    n_nulls_antes = df_macro.isnull().sum().sum()
    df_macro = df_macro.ffill().bfill()
    n_nulls_despues = df_macro.isnull().sum().sum()
    
    print(f"\nFeatures macro generadas:")
    print(f"- Columnas: {list(df_macro.columns)}")
    print(f"- Shape: {df_macro.shape}")
    print(f"- Valores nulos antes de fill: {n_nulls_antes}")
    print(f"- Valores nulos después de fill: {n_nulls_despues}")
    
    return df_macro


def agregar_google_trends(df, keywords=None, fecha_inicio=None, fecha_fin=None):
    """Google Trends con caché en CSV para evitar rate limiting"""
    ruta_cache = os.path.join('..', 'data', 'processed', 'google_trends.csv')
    if os.path.exists(ruta_cache):
        df_trends = pd.read_csv(ruta_cache, index_col=0, parse_dates=True)
        df_trends = df_trends.reindex(df.index).ffill().bfill()
        return df_trends
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords, timeframe=f'{fecha_inicio} {fecha_fin}')
        df_trends = pytrends.interest_over_time()
        if not df_trends.empty:
            df_trends = df_trends.drop(columns=['isPartial'], errors='ignore')
            df_trends_diario = df_trends.resample('D').interpolate('linear')
            df_trends_diario = df_trends_diario.reindex(df.index).ffill().bfill()
            delta_trends = df_trends_diario.diff()
            delta_trends.columns = [f"trends_delta_{col}" for col in delta_trends.columns]
            delta_trends.to_csv(ruta_cache)
            return delta_trends
    except Exception as e:
        print(f"   ⚠️ Google Trends no disponible: {e}")
    return pd.DataFrame(index=df.index)


def calcular_features_interaccion(df):
    """
    Crea features de interacción entre variables clave.

    Features creadas:
      - vol_x_vix: volatilidad_20d de SP500 × DELTA_VIX (pánico con alta vol)
      - mom_x_brent: momentum_5d de BRENT × retorno_BRENT (activo energético en tendencia)
      - rsi_extremo_*: 1 si RSI < 30 o RSI > 70 para cada activo

    Args:
        df (pandas.DataFrame): DataFrame con features base ya calculadas.

    Returns:
        pandas.DataFrame: DataFrame con columnas de interacción añadidas.

    Example:
        >>> df_interaccion = calcular_features_interaccion(df_features)
        >>> print(df_interaccion[['vol_x_vix', 'mom_x_brent']].head())
    """
    print("\n" + "="*80)
    print("CALCULANDO FEATURES DE INTERACCIÓN")
    print("="*80)

    df_result = df.copy()
    nuevas_columnas = []

    # 1. vol_x_vix: volatilidad de SP500 × cambio VIX
    vol_col = 'SP500_vol20'
    delta_vix_col = 'DELTA_VIX'
    if vol_col in df.columns and delta_vix_col in df.columns:
        nombre_inter = 'vol_x_vix'
        df_result[nombre_inter] = df[vol_col] * df[delta_vix_col]
        nuevas_columnas.append(nombre_inter)
        print(f"  ✓ Creada: {nombre_inter}")
    else:
        print(f"  ⚠ {nombre_inter}: columnas base no disponibles ({vol_col}, {delta_vix_col})")

    # 2. mom_x_brent: momentum de BRENT × retorno de BRENT
    mom_col = 'BRENT_mom5'
    ret_col = 'BRENT'
    if mom_col in df.columns and ret_col in df.columns:
        nombre_inter = 'mom_x_brent'
        df_result[nombre_inter] = df[mom_col] * df[ret_col]
        nuevas_columnas.append(nombre_inter)
        print(f"  ✓ Creada: {nombre_inter}")
    else:
        print(f"  ⚠ {nombre_inter}: columnas base no disponibles ({mom_col}, {ret_col})")

    # 3. rsi_extremo para cada activo con RSI
    for col in df.columns:
        if col.endswith('_rsi'):
            activo = col.replace('_rsi', '')
            nombre_extremo = f"{activo}_rsi_extremo"
            df_result[nombre_extremo] = ((df[col] < 30) | (df[col] > 70)).astype(int)
            nuevas_columnas.append(nombre_extremo)

    # Imprimir resumen
    print(f"\nFeatures de interacción creadas: {len(nuevas_columnas)} columnas")
    print(f"- Nuevas columnas: {nuevas_columnas}")
    print(f"- Shape resultante: {df_result.shape}")

    return df_result


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
    Incluye retornos lag1 de VIX, BRENT, MERVAL y GOLD como features adicionales.

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

    # Añadir retornos lag1 de VIX, BRENT, MERVAL y GOLD a df_features
    print("\nAñadiendo retornos lag1 (desfase 1 día)...")
    activos_lag1 = ['VIX', 'BRENT', 'MERVAL', 'GOLD']
    for activo in activos_lag1:
        if activo in df_retornos.columns:
            col_lag = f"{activo}_lag1"
            df_features[col_lag] = df_retornos[activo].shift(1)
            print(f"  ✓ Añadido: {col_lag}")
        else:
            print(f"  ⚠ {activo} no disponible en df_retornos")

    # Combinar retornos y features
    print("\nCombinando retornos y features...")
    df_combinado = pd.concat([df_retornos, df_features], axis=1)

    # Calcular features de interacción sobre el dataframe combinado
    print("\nCalculando features de interacción...")
    df_combinado = calcular_features_interaccion(df_combinado)

    # Añadir columna de sector
    print("\nAsignando sectores a cada activo...")
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

    # Listar todas las columnas
    print(f"\nColumnas del dataset ({len(df_combinado.columns)} total):")
    for i, col in enumerate(df_combinado.columns, 1):
        print(f"  {i:2d}. {col}")

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

        print("Retornos diarios cargados:")
        print(f"- Shape: {df_retornos.shape}")
        print(f"- Columnas: {list(df_retornos.columns)}")

        # =============================================================================
        # CALCULAR FEATURES TÉCNICAS
        # =============================================================================
        print("\n" + "="*80)
        print("CALCULANDO FEATURES TÉCNICAS")
        print("="*80)

        df_vol = calcular_volatilidad_historica(df_retornos, ventana=20)
        df_mom = calcular_momentum(df_retornos, ventana=5)
        df_corr = calcular_correlacion_rodante_brent(df_retornos, ventana=30)
        delta_vix = calcular_delta_vix(df_retornos)

        # Nuevas features técnicas
        df_rsi = calcular_rsi(df_retornos, ventana=14)
        df_bb = calcular_bandas_bollinger(df_retornos, ventana=20)
        df_ret_mensual = calcular_retorno_mensual(df_retornos, ventana=21)

        # Features de volumen (requiere datos de volumen - optional)
        # df_volumen = ... # Cargar datos de volumen si disponibles
        # df_ratio_vol = calcular_ratio_volumen(df_volumen, ventana=20)

        # =============================================================================
        # FEATURES MACROECONÓMICAS (opcional — requiere yfinance)
        # =============================================================================
        print("\n" + "="*80)
        print("FEATURES MACROECONÓMICAS (opcional)")
        print("="*80)

        fecha_inicio = '2020-01-01'
        fecha_fin = '2026-04-30'

        try:
            df_macro = agregar_features_macro(df_retornos, fecha_inicio, fecha_fin)
            print(f"✓ Features macro agregadas: {df_macro.shape[1]} columnas")
        except Exception as e:
            print(f"⚠️ No se pudieron descargar features macro: {e}")
            df_macro = pd.DataFrame(index=df_retornos.index)

        # =============================================================================
        # GOOGLE TRENDS (opcional — requiere pytrends)
        # =============================================================================
        try:
            df_trends = agregar_google_trends(df_retornos, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
            if not df_trends.empty:
                print(f"✓ Google Trends agregado: {df_trends.shape[1]} columnas")
            else:
                df_trends = pd.DataFrame(index=df_retornos.index)
        except Exception as e:
            print(f"⚠️ No se pudieron descargar Google Trends: {e}")
            df_trends = pd.DataFrame(index=df_retornos.index)

        # =============================================================================
        # COMBINAR TODAS LAS FEATURES
        # =============================================================================
        print("\n" + "="*80)
        print("COMBINANDO TODAS LAS FEATURES")
        print("="*80)

        # Concatenar todas las features (excepto macro y trends si están vacíos)
        features_list = [df_vol, df_mom, df_corr, delta_vix, df_rsi, df_bb, df_ret_mensual]

        if not df_macro.empty:
            features_list.append(df_macro)
        if not df_trends.empty:
            features_list.append(df_trends)

        # Convertir delta_vix a DataFrame para concatenar
        if isinstance(delta_vix, pd.Series):
            delta_vix = delta_vix.to_frame()

        df_features = pd.concat(features_list, axis=1)

        print(f"Features combinadas: {df_features.shape}")

        # Crear indicador de ventana
        df_features = crear_indicador_ventana(df_features, EVENT_DATE)

        # =============================================================================
        # CONSTRUIR DATASET FINAL
        # =============================================================================
        df_modelo = construir_dataset_modelamiento(df_retornos, df_features)

        # Guardar dataset
        ruta_guardado = os.path.join("data", "processed", "dataset_modelamiento.csv")
        df_modelo.to_csv(ruta_guardado)
        print(f"\nDataset guardado en: {ruta_guardado}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_datos}")
        print("Ejecute primero data_collection.py para generar los datos necesarios.")