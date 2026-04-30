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


# =============================================================================
# NUEVAS FUNCIONES DE FEATURE ENGINEERING
# =============================================================================

def calcular_rsi(df, ventana=14):
    """
    Calcula el Índice de Fuerza Relativa (RSI) de 14 días para cada activo.

    Args:
        df (pandas.DataFrame): DataFrame con retornos logarítmicos.
        ventana (int): Tamaño de la ventana para el cálculo (default 14).

    Returns:
        pandas.DataFrame: DataFrame con RSI para cada activo ( valor entre 0 y 100).

    Example:
        >>> df_rsi = calcular_rsi(df_retornos, ventana=14)
        >>> print(df_rsi.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO RSI (VENTANA {ventana} DÍAS)")
    print("="*80)

    df_rsi = pd.DataFrame(index=df.index)

    for columna in df.columns:
        # Calcular cambios diarios
        delta = df[columna].diff()

        # Separar gains y losses
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        # Calcular medias exponenciales (más común para RSI)
        avg_gain = gains.ewm(com=ventana-1, adjust=True).mean()
        avg_loss = losses.ewm(com=ventana-1, adjust=True).mean()

        # Calcular RS y RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df_rsi[f"{columna}_rsi"] = rsi

    # Imprimir resumen
    print(f"\nRSI calculado con ventana de {ventana} días")
    print(f"- Shape: {df_rsi.shape}")
    print(f"- Primeros valores disponibles desde: {df_rsi.dropna().index[0]}")
    print(f"- Valores nulos: {df_rsi.isnull().sum().sum()} (primeros {ventana} días)")
    print(f"- Rango de valores: [{df_rsi.min().min():.2f}, {df_rsi.max().max():.2f}]")

    return df_rsi


def calcular_bandas_bollinger(df, ventana=20):
    """
    Calcula la distancia del precio actual a la banda superior de Bollinger
    en desviaciones estándar (Z-score).

    Args:
        df (pandas.DataFrame): DataFrame con precios (close) o retornos.
        ventana (int): Tamaño de la ventana para la media móvil y desviación.

    Returns:
        pandas.DataFrame: DataFrame con el z-score de la banda superior.

    Example:
        >>> df_bb = calcular_bandas_bollinger(df_precios, ventana=20)
        >>> print(df_bb.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO BANDAS DE BOLLINGER (VENTANA {ventana} DÍAS)")
    print("="*80)

    df_bb = pd.DataFrame(index=df.index)

    for columna in df.columns:
        # Calcular media móvil y desviación estándar
        media_movil = df[columna].rolling(window=ventana).mean()
        std_movil = df[columna].rolling(window=ventana).std()

        # Banda superior = media + 2*std
        banda_superior = media_movil + 2 * std_movil

        # Z-score: (precio - media) / std
        z_score = (df[columna] - media_movil) / std_movil

        df_bb[f"{columna}_bb_zscore"] = z_score

    # Imprimir resumen
    print(f"\nBandas de Bollinger calculadas con ventana de {ventana} días")
    print(f"- Shape: {df_bb.shape}")
    print(f"- Primeros valores disponibles desde: {df_bb.dropna().index[0]}")
    print(f"- Valores nulos: {df_bb.isnull().sum().sum()} (primeros {ventana-1} días)")
    print(f"- Rango de z-score: [{df_bb.min().min():.2f}, {df_bb.max().max():.2f}]")
    print(f"  (valores > 2 indican precio por encima de banda superior)")

    return df_bb


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
    Calcula el retorno acumulado de los últimos N días hábiles
    (aproximadamente un mes trading).

    Args:
        df (pandas.DataFrame): DataFrame con retornos logarítmicos.
        ventana (int): Número de días hábiles (default 21).

    Returns:
        pandas.DataFrame: DataFrame con retorno acumulado en la ventana.

    Example:
        >>> df_ret_mensual = calcular_retorno_mensual(df_retornos, ventana=21)
        >>> print(df_ret_mensual.head())
    """
    print("\n" + "="*80)
    print(f"CALCULANDO RETORNO MENSUAL ({ventana} DÍAS HÁBILES)")
    print("="*80)

    df_ret_mensual = pd.DataFrame(index=df.index)

    for columna in df.columns:
        # Suma de retornos logarítmicos en la ventana
        retorno_acum = df[columna].rolling(window=ventana).sum()
        df_ret_mensual[f"{columna}_ret_{ventana}d"] = retorno_acum

    # Imprimir resumen
    print(f"\nRetorno mensual calculado con ventana de {ventana} días")
    print(f"- Shape: {df_ret_mensual.shape}")
    print(f"- Primeros valores disponibles desde: {df_ret_mensual.dropna().index[0]}")
    print(f"- Valores nulos: {df_ret_mensual.isnull().sum().sum()} (primeros {ventana-1} días)")
    print(f"- Rango de retornos: [{df_ret_mensual.min().min()*100:.2f}%, {df_ret_mensual.max().max()*100:.2f}%]")

    return df_ret_mensual


def agregar_features_macro(df, fecha_inicio, fecha_fin):
    """
    Descarga y agrega features macroeconómicas al DataFrame.

    Activos descargados desde Yahoo Finance:
      - DXY (índice del dólar): símbolo 'DX-Y.NYB'
      - Gas natural: 'NG=F'
      - Spread 10Y-2Y: ^TNX - ^IRX (ambos son yields de treasuries)
      - Oro/Petróleo ratio: GC=F / BZ=F

    Args:
        df (pandas.DataFrame): DataFrame con índice de fechas para alinear.
        fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        fecha_fin (str): Fecha de fin en formato 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: DataFrame con retornos logarítmicos de las features macro.

    Note:
        Los datos se descargan con yfinance. Los nulos se imputan con ffill().
    """
    print("\n" + "="*80)
    print("DESCARGANDO FEATURES MACROECONÓMICAS")
    print("="*80)

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance no está instalado. Ejecuta: pip install yfinance")
        return pd.DataFrame(index=df.index)

    features_macro = {}

    # 1. DXY - Índice del Dólar
    print("\n1. Descargando DXY (índice del dólar)...")
    try:
        dxy = yf.download('DX-Y.NYB', start=fecha_inicio, end=fecha_fin, progress=False)['Close']
        features_macro['DXY'] = dxy
        print(f"   ✓ DXY descargado: {len(dxy)} observaciones")
    except Exception as e:
        print(f"   ✗ Error descargando DXY: {e}")

    # 2. Gas Natural
    print("\n2. Descargando Gas Natural (NG=F)...")
    try:
        gas = yf.download('NG=F', start=fecha_inicio, end=fecha_fin, progress=False)['Close']
        features_macro['GAS'] = gas
        print(f"   ✓ Gas descargado: {len(gas)} observaciones")
    except Exception as e:
        print(f"   ✗ Error descargando Gas: {e}")

    # 3. Spread 10Y-2Y (diferencia de yields)
    print("\n3. Descargando Treasury yields (^TNX y ^IRX)...")
    try:
        tnx = yf.download('^TNX', start=fecha_inicio, end=fecha_fin, progress=False)['Close']
        irx = yf.download('^IRX', start=fecha_inicio, end=fecha_fin, progress=False)['Close']
        spread = tnx - irx
        spread.name = 'SPREAD_10Y2Y'
        features_macro['SPREAD_10Y2Y'] = spread
        print(f"   ✓ Spread calculado: {len(spread)} observaciones")
    except Exception as e:
        print(f"   ✗ Error descargando spreads: {e}")

    # 4. Oro/Petróleo ratio
    print("\n4. Descargando Oro (GC=F) y Petróleo (BZ=F)...")
    try:
        oro = yf.download('GC=F', start=fecha_inicio, end=fecha_fin, progress=False)['Close']
        brent = yf.download('BZ=F', start=fecha_inicio, end=fecha_fin, progress=False)['Close']
        ratio = oro / brent
        ratio.name = 'ORO_BRENT_RATIO'
        features_macro['ORO_BRENT_RATIO'] = ratio
        print(f"   ✓ Ratio calculado: {len(ratio)} observaciones")
    except Exception as e:
        print(f"   ✗ Error calculando ratio Oro/Petróleo: {e}")

    # Combinar todas las features macro en un DataFrame
    df_macro = pd.DataFrame(features_macro)

    # Calcular retornos logarítmicos
    df_macro_retornos = np.log(df_macro / df_macro.shift(1))

    # Alinear con el índice de df (usar reindex y ffill)
    df_macro_retornos = df_macro_retornos.reindex(df.index)
    df_macro_retornos = df_macro_retornos.ffill().bfill()

    # Imprimir resumen final
    print(f"\nFeatures macroeconómicas procesadas:")
    print(f"- Shape: {df_macro_retornos.shape}")
    print(f"- Columnas: {list(df_macro_retornos.columns)}")
    print(f"- Valores nulos después de ffill: {df_macro_retornos.isnull().sum().sum()}")

    return df_macro_retornos


def agregar_google_trends(df, keywords=None, fecha_inicio=None, fecha_fin=None):
    """
    Descarga datos de Google Trends para las keywords especificadas usando pytrends.

    Args:
        df (pandas.DataFrame): DataFrame con índice de fechas para alinear.
        keywords (list): Lista de términos de búsqueda (default: temas Venezuela).
        fecha_inicio (str): Fecha de inicio 'YYYY-MM-DD'.
        fecha_fin (str): Fecha de fin 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: DataFrame con índice de trends diario (interpolado).

    Note:
        Los datos de Google Trends son semanales. Se interpolan a diario.
        Si pytrends falla por rate limiting, se intenta cargar desde
        data/processed/google_trends.csv. Si no existe, se retorna DataFrame vacío.
    """
    print("\n" + "="*80)
    print("DESCARGANDO GOOGLE TRENDS")
    print("="*80)

    if keywords is None:
        keywords = ['Maduro', 'Venezuela oil', 'Venezuela crisis', 'Venezuela sanctions']

    # Ruta para cache de Google Trends
    ruta_cache = os.path.join('data', 'processed', 'google_trends.csv')

    # Intentar cargar desde cache primero
    if os.path.exists(ruta_cache):
        print(f"\nCargando Google Trends desde caché: {ruta_cache}")
        df_trends = pd.read_csv(ruta_cache, index_col=0, parse_dates=True)
        # Interpolar a diario
        df_trends_daily = df_trends.reindex(df.index).interpolate(method='linear').ffill().bfill()
        return df_trends_daily

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("ERROR: pytrends no está instalado. Ejecuta: pip install pytrends")
        return pd.DataFrame(index=df.index)

    try:
        # Inicializar pytrends
        pytrends = TrendReq(hl='es-US', tz=0)

        # Construir payload
        pytrends.build_payload(keywords, cat=0, timeframe=f'{fecha_inicio} {fecha_fin}', geo='')

        # Obtener datos de interés a lo largo del tiempo
        data = pytrends.interest_over_time()

        if data.empty:
            print("   ⚠️ No se obtuvieron datos de Google Trends")
            return pd.DataFrame(index=df.index)

        print(f"\n✓ Datos de Google Trends descargados: {len(data)} semanas")

        # Renombrar columnas
        data = data[keywords]  # Solo留下 las keywords
        data.index = pd.to_datetime(data.index)

        # Guardar en cache
        os.makedirs(os.path.dirname(ruta_cache), exist_ok=True)
        data.to_csv(ruta_cache)
        print(f"   ✓ Datos guardados en caché: {ruta_cache}")

        # Convertir de semanal a diario: interpolar linealmente
        print("\nConvirtiendo datos semanales a diarios...")
        data_daily = data.resample('D').interpolate(method='linear')
        data_daily = data_daily.ffill().bfill()

        # Calcular delta diario (cambio diario en interés)
        delta_trends = data_daily.diff(1)
        delta_trends.columns = [f"{col}_trend_delta" for col in delta_trends.columns]

        # Alinear con df
        delta_trends = delta_trends.reindex(df.index).ffill().bfill()

        print(f"✓ Trends processados: {delta_trends.shape}")
        print(f"- Columnas: {list(delta_trends.columns)}")

        return delta_trends

    except Exception as e:
        print(f"Error descargando Google Trends: {e}")
        print(" Continuando sin esta feature...")
        return pd.DataFrame(index=df.index)


def calcular_features_interaccion(df):
    """
    Crea features de interacción entre variables clave.

    Features creadas:
      - volatilidad_x_vix: volatilidad_20d * delta_vix (pánico con alta vol)
      - momentum_x_brent: momentum_5d * correlacion_brent_30d (activos energéticos en tendencia)
      - rsi_extremo: 1 si RSI < 30 (sobreventa) o RSI > 70 (sobrecompra), 0 en caso contrario

    Args:
        df (pandas.DataFrame): DataFrame con features base ya calculadas.

    Returns:
        pandas.DataFrame: DataFrame con columnas de interacción añadidas.

    Example:
        >>> df_interaccion = calcular_features_interaccion(df_features)
        >>> print(df_interaccion[['volatilidad_x_vix', 'rsi_extremo']].head())
    """
    print("\n" + "="*80)
    print("CALCULANDO FEATURES DE INTERACCIÓN")
    print("="*80)

    df_result = df.copy()
    nuevas_columnas = []

    # 1. volatilidad_x_vix: combina volatilidad propia con cambio en VIX
    # Buscar columnas de volatilidad (terminan en _vol20)
    for col in df.columns:
        if col.endswith('_vol20'):
            activo = col.replace('_vol20', '')
            delta_vix_col = 'DELTA_VIX'
            if delta_vix_col in df.columns:
                nombre_inter = f"{activo}_vol_x_vix"
                df_result[nombre_inter] = df[col] * df[delta_vix_col]
                nuevas_columnas.append(nombre_inter)

    # 2. momentum_x_brent: momentum del activo × su correlación con Brent
    for col in df.columns:
        if col.endswith('_mom5'):
            activo = col.replace('_mom5', '')
            corr_col = f"{activo}_corr_brent"
            if corr_col in df.columns:
                nombre_inter = f"{activo}_mom_x_corr"
                df_result[nombre_inter] = df[col] * df[corr_col]
                nuevas_columnas.append(nombre_inter)

    # 3. rsi_extremo: indicator binario de sobrecompra/sobreventa
    for col in df.columns:
        if col.endswith('_rsi'):
            activo = col.replace('_rsi', '')
            nombre_extremo = f"{activo}_rsi_extremo"
            # 1 si RSI < 30 o RSI > 70, 0 en otro caso
            df_result[nombre_extremo] = ((df[col] < 30) | (df[col] > 70)).astype(int)
            nuevas_columnas.append(nombre_extremo)

    # Imprimir resumen
    print(f"\nFeatures de interacción creadas:")
    print(f"- Nuevas columnas: {len(nuevas_columnas)}")
    for col in nuevas_columnas:
        print(f"  • {col}")
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

    # Calcular features de interacción a partir de df_features
    print("\nCalculando features de interacción...")
    df_features = calcular_features_interaccion(df_features)

    # Combinar retornos y features
    df_combinado = pd.concat([df_retornos, df_features], axis=1)

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