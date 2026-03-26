#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para el estudio de eventos financieros.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import os
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

# Constantes
VENTANA_ESTIMACION_INICIO = -250
VENTANA_ESTIMACION_FIN = -11
VENTANA_EVENTO_INICIO = -10
VENTANA_EVENTO_FIN = 60


def estimar_modelo_mercado(df_retornos, activo, mercado='SP500', inicio_est=-250, fin_est=-11, event_date=None):
    """
    Estima los parámetros del modelo de mercado: R_activo = α + β × R_mercado.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        activo (str): Nombre del activo a modelar.
        mercado (str): Nombre del activo de referencia (mercado).
        inicio_est (int): Inicio de la ventana de estimación (días relativos al evento).
        fin_est (int): Fin de la ventana de estimación (días relativos al evento).
        event_date (str or datetime): Fecha del evento.
    
    Returns:
        dict: Diccionario con parámetros estimados {'alpha': α, 'beta': β, 'r2': R²}.
    
    Example:
        >>> params = estimar_modelo_mercado(df_retornos, 'BRENT', 'SP500', -250, -11, '2026-01-03')
        >>> print(f"Alpha: {params['alpha']:.4f}, Beta: {params['beta']:.4f}, R²: {params['r2']:.4f}")
    """
    # Convertir event_date a datetime si es string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Verificar que los activos existan en el DataFrame
    if activo not in df_retornos.columns:
        raise ValueError(f"El activo '{activo}' no existe en el DataFrame")
    if mercado not in df_retornos.columns:
        raise ValueError(f"El mercado '{mercado}' no existe en el DataFrame")
    
    # Crear columna de días relativos al evento
    df_temp = df_retornos.copy()
    df_temp['dias_al_evento'] = (df_temp.index - event_date).days
    
    # Filtrar la ventana de estimación
    df_estimacion = df_temp[(df_temp['dias_al_evento'] >= inicio_est) & 
                            (df_temp['dias_al_evento'] <= fin_est)].copy()
    
    # Verificar que hay suficientes datos
    if len(df_estimacion) < 30:
        logger.warning(f"Pocos datos para estimar modelo de {activo}: {len(df_estimacion)} observaciones")
    
    # Estimar modelo con OLS
    X = sm.add_constant(df_estimacion[mercado])
    modelo = sm.OLS(df_estimacion[activo], X).fit()
    
    # Extraer parámetros
    alpha = modelo.params[0]
    beta = modelo.params[1]
    r2 = modelo.rsquared
    
    # Imprimir resultados
    print(f"\nModelo de mercado para {activo}:")
    print(f"- Alpha (α): {alpha:.6f}")
    print(f"- Beta (β): {beta:.6f}")
    print(f"- R²: {r2:.6f}")
    print(f"- Observaciones: {len(df_estimacion)}")
    
    return {'alpha': alpha, 'beta': beta, 'r2': r2, 'modelo': modelo}


def calcular_retorno_anormal(df_retornos, activo, params_modelo, mercado='SP500'):
    """
    Calcula los retornos anormales para un activo.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        activo (str): Nombre del activo.
        params_modelo (dict): Diccionario con parámetros del modelo {'alpha': α, 'beta': β}.
        mercado (str): Nombre del activo de referencia (mercado).
    
    Returns:
        pandas.Series: Serie con retornos anormales.
    
    Example:
        >>> ar = calcular_retorno_anormal(df_retornos, 'BRENT', params_modelo, 'SP500')
        >>> print(ar.head())
    """
    # Extraer parámetros
    alpha = params_modelo['alpha']
    beta = params_modelo['beta']
    
    # Calcular retornos esperados según el modelo de mercado
    retornos_esperados = alpha + beta * df_retornos[mercado]
    
    # Calcular retornos anormales (AR = retorno real - retorno esperado)
    ar = df_retornos[activo] - retornos_esperados
    ar.name = f'AR_{activo}'
    
    return ar


def calcular_car(serie_ar, inicio_ventana, fin_ventana, event_date):
    """
    Calcula el Cumulative Abnormal Return (CAR) en una ventana específica.
    
    Args:
        serie_ar (pandas.Series): Serie con retornos anormales.
        inicio_ventana (int): Inicio de la ventana (días relativos al evento).
        fin_ventana (int): Fin de la ventana (días relativos al evento).
        event_date (str or datetime): Fecha del evento.
    
    Returns:
        float: Valor del CAR en la ventana especificada.
    
    Example:
        >>> car = calcular_car(ar_brent, -5, 5, '2026-01-03')
        >>> print(f"CAR[-5,+5]: {car:.4f}")
    """
    # Convertir event_date a datetime si es string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Crear columna de días relativos al evento
    df_temp = pd.DataFrame(serie_ar)
    df_temp['dias_al_evento'] = (df_temp.index - event_date).days
    
    # Filtrar la ventana
    df_ventana = df_temp[(df_temp['dias_al_evento'] >= inicio_ventana) & 
                         (df_temp['dias_al_evento'] <= fin_ventana)]
    
    # Calcular CAR
    car = df_ventana[serie_ar.name].sum()
    
    return car


def test_significancia_ar(serie_ar, ventana_estimacion):
    """
    Aplica t-test sobre los AR en la ventana del evento.
    
    Args:
        serie_ar (pandas.Series): Serie con retornos anormales.
        ventana_estimacion (pandas.Series): Serie con retornos anormales en la ventana de estimación.
    
    Returns:
        dict: Diccionario con resultados del test {'t_stat': t, 'p_valor': p, 'significativo': bool}.
    
    Example:
        >>> resultado = test_significancia_ar(ar_evento, ar_estimacion)
        >>> print(f"t-stat: {resultado['t_stat']:.4f}, p-valor: {resultado['p_valor']:.4f}")
    """
    # Calcular estadísticos de la ventana de estimación
    media_estimacion = ventana_estimacion.mean()
    std_estimacion = ventana_estimacion.std()
    
    # Aplicar t-test (H0: AR promedio = 0)
    t_stat = (serie_ar.mean() - 0) / (std_estimacion / np.sqrt(len(serie_ar)))
    p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(serie_ar)-1))
    
    # Determinar significancia
    significativo = p_valor < 0.05
    
    return {
        't_stat': t_stat,
        'p_valor': p_valor,
        'significativo': significativo
    }


def calcular_ar_todos_activos(df_retornos, event_date):
    """
    Calcula los retornos anormales para todos los activos excepto el mercado.
    
    Args:
        df_retornos (pandas.DataFrame): DataFrame con retornos logarítmicos.
        event_date (str or datetime): Fecha del evento.
    
    Returns:
        pandas.DataFrame: DataFrame con retornos anormales para cada activo.
    
    Example:
        >>> df_ar = calcular_ar_todos_activos(df_retornos, '2026-01-03')
        >>> print(df_ar.head())
    """
    print("\n" + "="*80)
    print("CALCULANDO RETORNOS ANORMALES PARA TODOS LOS ACTIVOS")
    print("="*80)
    
    # Convertir event_date a datetime si es string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Crear DataFrame para almacenar los AR
    df_ar = pd.DataFrame(index=df_retornos.index)
    
    # Calcular AR para cada activo excepto SP500 (mercado)
    for activo in df_retornos.columns:
        if activo != 'SP500':
            print(f"\nProcesando {activo}...")
            
            # Estimar modelo de mercado
            params = estimar_modelo_mercado(df_retornos, activo, 'SP500', 
                                           VENTANA_ESTIMACION_INICIO, 
                                           VENTANA_ESTIMACION_FIN, 
                                           event_date)
            
            # Calcular retornos anormales
            ar = calcular_retorno_anormal(df_retornos, activo, params)
            
            # Añadir al DataFrame
            df_ar[f'AR_{activo}'] = ar
            
            # Calcular CAR en diferentes ventanas
            car_pre = calcular_car(ar, -5, -1, event_date)
            car_evento = calcular_car(ar, 0, 5, event_date)
            car_post = calcular_car(ar, 6, 20, event_date)
            
            print(f"- CAR[-5,-1]: {car_pre:.6f}")
            print(f"- CAR[0,+5]: {car_evento:.6f}")
            print(f"- CAR[+6,+20]: {car_post:.6f}")
    
    # Imprimir resumen
    print("\nRESUMEN DE RETORNOS ANORMALES:")
    print(f"- Activos procesados: {len(df_ar.columns)}")
    print(f"- Período: {df_ar.index.min()} a {df_ar.index.max()}")
    
    return df_ar


def crear_variable_objetivo(df_ar):
    """
    Crea variables objetivo binarias basadas en los retornos anormales.
    
    Args:
        df_ar (pandas.DataFrame): DataFrame con retornos anormales.
    
    Returns:
        pandas.DataFrame: DataFrame con variables objetivo.
    
    Example:
        >>> df_target = crear_variable_objetivo(df_ar)
        >>> print(df_target.head())
    """
    print("\n" + "="*80)
    print("CREANDO VARIABLES OBJETIVO")
    print("="*80)
    
    df_target = pd.DataFrame(index=df_ar.index)
    
    for columna in df_ar.columns:
        activo = columna.replace('AR_', '')
        nombre_target = f'target_{activo}'
        
        # Target = 1 si AR > 0, else 0
        df_target[nombre_target] = (df_ar[columna] > 0).astype(int)
        
        # Imprimir distribución de clases
        n_positivos = df_target[nombre_target].sum()
        n_negativos = len(df_target) - n_positivos
        pct_positivos = n_positivos / len(df_target) * 100
        
        print(f"\nDistribución para {activo}:")
        print(f"- Clase 1 (AR > 0): {n_positivos} ({pct_positivos:.2f}%)")
        print(f"- Clase 0 (AR ≤ 0): {n_negativos} ({100-pct_positivos:.2f}%)")
    
    return df_target


def graficar_car_activos(df_ar, event_date, lista_activos=None):
    """
    Genera gráficos de CAR acumulados para los activos seleccionados.
    
    Args:
        df_ar (pandas.DataFrame): DataFrame con retornos anormales.
        event_date (str or datetime): Fecha del evento.
        lista_activos (list): Lista de activos a graficar. Si es None, usa todos.
    
    Returns:
        matplotlib.figure.Figure: Figura con los gráficos.
    
    Example:
        >>> fig = graficar_car_activos(df_ar, '2026-01-03', ['BRENT', 'WTI', 'GOLD'])
        >>> plt.show()
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICOS DE CAR ACUMULADOS")
    print("="*80)
    
    # Convertir event_date a datetime si es string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Si no se especifica lista de activos, usar todos
    if lista_activos is None:
        lista_activos = [col.replace('AR_', '') for col in df_ar.columns]
    
    # Crear columna de días relativos al evento
    df_temp = df_ar.copy()
    df_temp['dias_al_evento'] = (df_temp.index - event_date).days
    
    # Filtrar período de interés (desde inicio estimación hasta fin evento)
    df_plot = df_temp[(df_temp['dias_al_evento'] >= VENTANA_ESTIMACION_INICIO) & 
                      (df_temp['dias_al_evento'] <= VENTANA_EVENTO_FIN)]
    
    # Ordenar por días al evento
    df_plot = df_plot.sort_values('dias_al_evento')
    
    # Calcular CAR acumulado para cada activo
    for activo in lista_activos:
        col_ar = f'AR_{activo}'
        if col_ar in df_plot.columns:
            df_plot[f'CAR_{activo}'] = df_plot[col_ar].cumsum()
    
    # Crear directorio para gráficos si no existe
    directorio = os.path.join("data", "processed", "graficos")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Crear figura
    n_activos = len(lista_activos)
    fig, axes = plt.subplots(n_activos, 1, figsize=(12, 4*n_activos), sharex=True)
    
    # Si solo hay un activo, convertir axes en lista
    if n_activos == 1:
        axes = [axes]
    
    # Graficar cada activo
    for i, activo in enumerate(lista_activos):
        col_car = f'CAR_{activo}'
        if col_car in df_plot.columns:
            ax = axes[i]
            
            # Graficar CAR
            ax.plot(df_plot['dias_al_evento'], df_plot[col_car], 'b-', linewidth=2)
            
            # Calcular bandas de confianza (±1.96 × std_AR)
            std_ar = df_plot[f'AR_{activo}'].std()
            df_plot[f'upper_{activo}'] = df_plot[col_car] + 1.96 * std_ar * np.sqrt(np.abs(df_plot['dias_al_evento']))
            df_plot[f'lower_{activo}'] = df_plot[col_car] - 1.96 * std_ar * np.sqrt(np.abs(df_plot['dias_al_evento']))
            
            # Graficar bandas de confianza
            ax.fill_between(df_plot['dias_al_evento'], 
                           df_plot[f'lower_{activo}'], 
                           df_plot[f'upper_{activo}'], 
                           color='b', alpha=0.1)
            
            # Sombrear área de la ventana del evento
            ax.axvspan(VENTANA_EVENTO_INICIO, VENTANA_EVENTO_FIN, 
                      alpha=0.2, color='gray', label='Ventana del evento')
            
            # Línea vertical en el día del evento
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                      label='Fecha del evento')
            
            # Línea horizontal en y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Configurar gráfico
            ax.set_title(f'CAR Acumulado - {activo}', fontsize=14)
            ax.set_ylabel('CAR', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Anotar valores clave
            car_evento = df_plot.loc[df_plot['dias_al_evento'] == 5, col_car].values[0] if 5 in df_plot['dias_al_evento'].values else None
            if car_evento is not None:
                ax.annotate(f'CAR[0,+5] = {car_evento:.4f}', 
                           xy=(5, car_evento),
                           xytext=(10, 0), 
                           textcoords='offset points',
                           fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    # Configurar eje x compartido
    axes[-1].set_xlabel('Días relativos al evento', fontsize=12)
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Guardar figura
    ruta_guardado = os.path.join(directorio, "car_activos.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {ruta_guardado}")
    
    return fig


if __name__ == "__main__":
    # Ejemplo de uso del módulo
    import os
    from data_collection import EVENT_DATE
    
    # Cargar datos de ejemplo
    try:
        ruta_datos = os.path.join("data", "processed", "retornos_diarios.csv")
        df_retornos = pd.read_csv(ruta_datos, index_col=0, parse_dates=True)
        
        # Calcular retornos anormales
        df_ar = calcular_ar_todos_activos(df_retornos, EVENT_DATE)
        
        # Crear variable objetivo
        df_target = crear_variable_objetivo(df_ar)
        
        # Graficar CAR
        fig = graficar_car_activos(df_ar, EVENT_DATE)
        
        # Guardar resultados
        ruta_ar = os.path.join("data", "processed", "retornos_anormales.csv")
        ruta_target = os.path.join("data", "processed", "variables_objetivo.csv")
        
        df_ar.to_csv(ruta_ar)
        df_target.to_csv(ruta_target)
        
        print(f"\nRetornos anormales guardados en: {ruta_ar}")
        print(f"Variables objetivo guardadas en: {ruta_target}")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_datos}")
        print("Ejecute primero data_collection.py para generar los datos necesarios.")