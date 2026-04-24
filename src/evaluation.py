#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la evaluación de modelos predictivos.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.inspection import permutation_importance
import os
import logging
from tabulate import tabulate
import joblib

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)


def calcular_metricas_completas(modelo, X_test, y_test, nombre_modelo, y_pred=None):
    """
    Calcula métricas completas de evaluación para un modelo.
    
    Args:
        modelo (estimator): Modelo entrenado.
        X_test (pandas.DataFrame): Features de prueba.
        y_test (pandas.Series): Variable objetivo de prueba.
        nombre_modelo (str): Nombre del modelo.
        y_pred (numpy.array, optional): Predicciones manuales. Si se pasa,
            se usa en lugar de recalcular con modelo.predict().
    
    Returns:
        dict: Diccionario con métricas e interpretaciones.
    
    Example:
        >>> metricas = calcular_metricas_completas(modelo, X_test, y_test, 'XGBoost')
        >>> print(metricas['auc'])
    """
    print("\n" + "="*80)
    print(f"EVALUACIÓN COMPLETA DEL MODELO: {nombre_modelo}")
    print("="*80)
    
    # Predecir probabilidades y clases
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Usar y_pred externo si se proporciona, sino predecir
    if y_pred is None:
        y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Generar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Imprimir métricas con interpretaciones
    print("\nMÉTRICAS DE EVALUACIÓN:")
    
    print(f"\n1. AUC-ROC: {auc:.4f}")
    interpretacion_auc = (f"AUC-ROC de {auc:.2f}: el modelo distingue correctamente entre subida "
                         f"y bajada en el {auc*100:.0f}% de los casos, superando en "
                         f"{(auc-0.5)*100:.0f} puntos porcentuales la línea base aleatoria de 0.50.")
    print(f"   Interpretación: {interpretacion_auc}")
    
    print(f"\n2. F1-Score: {f1:.4f}")
    interpretacion_f1 = (f"F1-Score de {f1:.2f}: balance adecuado entre no perderse subidas "
                        f"reales y no generar falsas alarmas de subida.")
    print(f"   Interpretación: {interpretacion_f1}")
    
    print(f"\n3. Accuracy: {accuracy:.4f}")
    interpretacion_acc = (f"Accuracy de {accuracy:.2f}: el modelo clasificó correctamente el "
                         f"{accuracy*100:.0f}% de los días del conjunto de prueba.")
    print(f"   Interpretación: {interpretacion_acc}")
    
    print(f"\n4. Precisión: {precision:.4f}")
    interpretacion_prec = (f"Precisión de {precision:.2f}: de cada 10 días predichos como subida, "
                          f"aproximadamente {precision*10:.0f} realmente subieron.")
    print(f"   Interpretación: {interpretacion_prec}")
    
    print(f"\n5. Recall: {recall:.4f}")
    interpretacion_rec = (f"Recall de {recall:.2f}: el modelo detectó el {recall*100:.0f}% de todos "
                         f"los días que realmente tuvieron retorno anormal positivo.")
    print(f"   Interpretación: {interpretacion_rec}")
    
    # Imprimir matriz de confusión
    print("\nMATRIZ DE CONFUSIÓN:")
    print("                  Predicción")
    print("                  Bajada  Subida")
    print(f"Real     Bajada   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"         Subida   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Calcular métricas adicionales de la matriz
    tn, fp, fn, tp = cm.ravel()
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nVerdaderos Positivos (TP): {tp}")
    print(f"Falsos Positivos (FP): {fp}")
    print(f"Verdaderos Negativos (TN): {tn}")
    print(f"Falsos Negativos (FN): {fn}")
    print(f"Especificidad: {especificidad:.4f}")
    
    # Almacenar métricas e interpretaciones
    metricas = {
        'auc': auc,
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'especificidad': especificidad,
        'matriz_confusion': cm,
        'interpretacion_auc': interpretacion_auc,
        'interpretacion_f1': interpretacion_f1,
        'interpretacion_acc': interpretacion_acc,
        'interpretacion_prec': interpretacion_prec,
        'interpretacion_rec': interpretacion_rec
    }
    
    return metricas


def graficar_curvas_roc(modelos_evaluados, X_test, y_test):
    """
    Genera gráfico de curvas ROC para varios modelos.
    
    Args:
        modelos_evaluados (dict): Diccionario con modelos entrenados.
        X_test (pandas.DataFrame): Features de prueba.
        y_test (pandas.Series): Variable objetivo de prueba.
    
    Returns:
        matplotlib.figure.Figure: Figura con las curvas ROC.
    
    Example:
        >>> fig = graficar_curvas_roc(modelos, X_test, y_test)
        >>> plt.show()
    """
    print("\n" + "="*80)
    print("GENERANDO CURVAS ROC")
    print("="*80)
    
    # Crear directorio para gráficos si no existe
    directorio = os.path.join("data", "processed", "graficos")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    
    # Colores para cada modelo
    colores = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    
    # Graficar línea de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio (AUC = 0.5)')
    
    # Graficar curva ROC para cada modelo
    for i, (nombre_modelo, modelo) in enumerate(modelos_evaluados.items()):
        # Predecir probabilidades
        y_proba = modelo.predict_proba(X_test)[:, 1]
        
        # Calcular AUC
        auc = roc_auc_score(y_test, y_proba)
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        # Graficar curva
        plt.plot(fpr, tpr, color=colores[i % len(colores)], lw=2,
                label=f'{nombre_modelo} (AUC = {auc:.3f})')
    
    # Configurar gráfico
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC de los Modelos', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    ruta_guardado = os.path.join(directorio, "curvas_roc.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {ruta_guardado}")
    
    return plt.gcf()


def graficar_importancia_variables(pipeline_final, nombres_features):
    """
    Genera gráfico de importancia de variables para el modelo final.
    
    Args:
        pipeline_final (sklearn.pipeline.Pipeline): Pipeline con el modelo final.
        nombres_features (list): Lista con nombres de las features.
    
    Returns:
        matplotlib.figure.Figure: Figura con la importancia de variables.
    
    Example:
        >>> fig = graficar_importancia_variables(pipeline, X_train.columns)
        >>> plt.show()
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO DE IMPORTANCIA DE VARIABLES")
    print("="*80)
    
    # Crear directorio para gráficos si no existe
    directorio = os.path.join("data", "processed", "graficos")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Extraer el modelo del pipeline
    modelo = pipeline_final.named_steps['modelo']
    
    # Determinar el tipo de modelo
    importancias = None
    
    # Para modelos basados en árboles
    if hasattr(modelo, 'feature_importances_'):
        print("Extrayendo feature importances del modelo basado en árboles...")
        importancias = modelo.feature_importances_
        metodo = "feature_importances_"
    
    # Para otros modelos, usar permutation importance
    else:
        print("Calculando permutation importance...")
        # Asumimos que X_val es un conjunto de validación disponible
        # En un caso real, deberíamos tener estos datos disponibles
        # o usar una muestra del conjunto de entrenamiento
        X_sample = X_val[:100] if 'X_val' in globals() else X_train[:100]
        y_sample = y_val[:100] if 'y_val' in globals() else y_train[:100]
        
        result = permutation_importance(
            pipeline_final, X_sample, y_sample, n_repeats=10, random_state=42
        )
        importancias = result.importances_mean
        metodo = "permutation_importance"
    
    # Crear DataFrame con importancias
    df_importancias = pd.DataFrame({
        'feature': nombres_features,
        'importancia': importancias
    })
    
    # Ordenar por importancia
    df_importancias = df_importancias.sort_values('importancia', ascending=False)
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar barras horizontales
    sns.barplot(x='importancia', y='feature', data=df_importancias, palette='viridis')
    
    # Configurar gráfico
    plt.title(f'Importancia de Variables ({metodo})', fontsize=14)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)
    
    # Guardar figura
    ruta_guardado = os.path.join(directorio, "importancia_variables.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {ruta_guardado}")
    
    return plt.gcf()


def comparar_modelos_tabla(resultados_todos_modelos):
    """
    Genera tabla comparativa de todos los modelos evaluados.
    
    Args:
        resultados_todos_modelos (dict): Diccionario con resultados de todos los modelos.
    
    Returns:
        pandas.DataFrame: DataFrame con la comparación.
    
    Example:
        >>> df_comparacion = comparar_modelos_tabla(resultados)
        >>> print(df_comparacion)
    """
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)
    
    # Crear DataFrame para la comparación
    df_comparacion = pd.DataFrame(columns=['Modelo', 'AUC-ROC', 'F1-Score', 'Accuracy', 'Precision', 'Recall'])
    
    # Llenar DataFrame con resultados
    for nombre_modelo, metricas in resultados_todos_modelos.items():
        df_comparacion = df_comparacion.append({
            'Modelo': nombre_modelo,
            'AUC-ROC': metricas['auc'],
            'F1-Score': metricas['f1'],
            'Accuracy': metricas['accuracy'],
            'Precision': metricas['precision'],
            'Recall': metricas['recall']
        }, ignore_index=True)
    
    # Encontrar el mejor valor por columna
    mejores_valores = {}
    for col in df_comparacion.columns:
        if col != 'Modelo':
            mejores_valores[col] = df_comparacion[col].max()
    
    # Crear tabla para imprimir con formato
    tabla = []
    for _, fila in df_comparacion.iterrows():
        tabla_fila = [fila['Modelo']]
        
        for col in ['AUC-ROC', 'F1-Score', 'Accuracy', 'Precision', 'Recall']:
            valor = fila[col]
            if valor == mejores_valores[col]:
                # Resaltar el mejor valor
                tabla_fila.append(f"\033[92m{valor:.4f}\033[0m")  # Verde en consola
            else:
                tabla_fila.append(f"{valor:.4f}")
        
        tabla.append(tabla_fila)
    
    # Imprimir tabla
    headers = ['Modelo', 'AUC-ROC', 'F1-Score', 'Accuracy', 'Precision', 'Recall']
    print(tabulate(tabla, headers=headers, tablefmt="grid"))
    
    return df_comparacion


def interpretar_resultados_evento(df_ar, event_date, activos):
    """
    Interpreta los resultados del estudio de eventos para cada activo.
    
    Args:
        df_ar (pandas.DataFrame): DataFrame con retornos anormales.
        event_date (str or datetime): Fecha del evento.
        activos (list): Lista de activos a interpretar.
    
    Returns:
        dict: Diccionario con interpretaciones por activo.
    
    Example:
        >>> interpretaciones = interpretar_resultados_evento(df_ar, '2026-01-03', ['BRENT', 'WTI'])
        >>> print(interpretaciones['BRENT'])
    """
    print("\n" + "="*80)
    print("INTERPRETACIÓN DE RESULTADOS DEL EVENTO")
    print("="*80)
    
    # Convertir event_date a datetime si es string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Crear columna de días relativos al evento
    df_temp = df_ar.copy()
    df_temp['dias_al_evento'] = (df_temp.index - event_date).days
    
    # Diccionario para almacenar interpretaciones
    interpretaciones = {}
    
    # Para cada activo
    for activo in activos:
        col_ar = f'AR_{activo}'
        
        if col_ar not in df_temp.columns:
            print(f"El activo {activo} no tiene retornos anormales calculados")
            continue
        
        print(f"\nAnálisis para {activo}:")
        
        # Filtrar ventana [-5, +5]
        df_ventana = df_temp[(df_temp['dias_al_evento'] >= -5) & 
                            (df_temp['dias_al_evento'] <= 5)]
        
        # Calcular CAR
        car = df_ventana[col_ar].sum()
        
        # Realizar t-test
        # Ventana de estimación para comparación
        df_estimacion = df_temp[(df_temp['dias_al_evento'] >= -250) & 
                               (df_temp['dias_al_evento'] <= -11)]
        
        # Calcular estadísticos
        media_estimacion = df_estimacion[col_ar].mean()
        std_estimacion = df_estimacion[col_ar].std()
        n = len(df_ventana)
        
        # t-test
        t_stat = (car - 0) / (std_estimacion / np.sqrt(n))
        p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        significativo = p_valor < 0.05
        
        # Imprimir resultados
        print(f"- CAR[-5, +5]: {car:.6f} ({car*100:.2f}%)")
        print(f"- Estadísticamente significativo: {'Sí' if significativo else 'No'} (p={p_valor:.4f})")
        
        # Generar interpretación
        direccion = "positivo" if car > 0 else "negativo"
        magnitud = abs(car) * 100
        
        if significativo:
            interpretacion = (
                f"El {activo} generó un retorno anormal acumulado de {direccion} "
                f"{magnitud:.1f}% en los 5 días alrededor del evento, estadísticamente "
                f"significativo (p={p_valor:.3f}). Esto indica que la captura de Maduro "
                f"{'impulsó' if car > 0 else 'afectó negativamente'} el precio del activo "
                f"por {'encima' if car > 0 else 'debajo'} de su comportamiento histórico esperado."
            )
        else:
            interpretacion = (
                f"El {activo} generó un retorno anormal acumulado de {direccion} "
                f"{magnitud:.1f}% en los 5 días alrededor del evento, pero no es "
                f"estadísticamente significativo (p={p_valor:.3f}). Esto sugiere que "
                f"la captura de Maduro no tuvo un impacto claro en este activo más allá "
                f"de su comportamiento histórico esperado."
            )
        
        print(f"- Interpretación: {interpretacion}")
        
        # Almacenar interpretación
        interpretaciones[activo] = {
            'car': car,
            'p_valor': p_valor,
            'significativo': significativo,
            'interpretacion': interpretacion
        }
    
    return interpretaciones


if __name__ == "__main__":
    # Ejemplo de uso del módulo
    import os
    from data_collection import EVENT_DATE
    from sklearn.model_selection import train_test_split
    
    try:
        # Cargar datos
        ruta_datos = os.path.join("data", "processed", "dataset_modelamiento.csv")
        df = pd.read_csv(ruta_datos, index_col=0)
        
        # Cargar retornos anormales
        ruta_ar = os.path.join("data", "processed", "retornos_anormales.csv")
        df_ar = pd.read_csv(ruta_ar, index_col=0, parse_dates=True)
        
        # Cargar modelo
        ruta_modelo = os.path.join("models", "modelo_final.pkl")
        pipeline = joblib.load(ruta_modelo)
        
        # Ejemplo con BRENT como target
        X = df.drop(['target_BRENT'], axis=1)
        y = df['target_BRENT']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        
        # Calcular métricas
        metricas = calcular_metricas_completas(pipeline, X_test, y_test, 'Modelo Final')
        
        # Interpretar resultados del evento
        activos = ['BRENT', 'WTI', 'GOLD', 'EXXON', 'CHEVRON']
        interpretaciones = interpretar_resultados_evento(df_ar, EVENT_DATE, activos)
        
        print("\nEvaluación completada con éxito.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ejecute primero los módulos anteriores para generar los datos necesarios.")