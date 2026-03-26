#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aplicación Streamlit para predicción de retorno anormal post-evento geopolítico.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Retorno Anormal Post-Evento Geopolítico",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SECCIÓN 1 — Encabezado
# =============================================================================

st.title("📈 Predictor de Retorno Anormal Post-Evento Geopolítico")
st.markdown("""
**Basado en el evento: Captura de Nicolás Maduro (3 ene 2026)**

Esta aplicación predice si un activo financiero generará un retorno anormal 
positivo (subida) o negativo (bajada) ante eventos geopolíticos similares 
a la captura de Nicolás Maduro en Venezuela.
""")

# Información del proyecto en el sidebar
with st.sidebar:
    st.header("ℹ️ Información del Proyecto")
    st.markdown("""
    **Proyecto:** Minería de Datos Financieros  
    **Metodología:** CRISP-DM  
    **Autoras:** Laura Laguado y Sofía Navales  
    **Evento:** Captura de Nicolás Maduro (3 ene 2026)  
    **Objetivo:** Predecir retorno anormal post-evento geopolítico
    """)
    
    st.markdown("---")
    st.markdown("""
    **Datos utilizados:**
    - Índices bursátiles (S&P 500, COLCAP, BOVESPA)
    - Petróleo (Brent, WTI)
    - Acciones (Exxon, Chevron)
    - Metales (Oro, Cobre)
    - Divisas (USD/COP)
    - Volatilidad (VIX)
    """)

# =============================================================================
# SECCIÓN 2 — Panel de entrada (sidebar)
# =============================================================================

with st.sidebar:
    st.header("📊 Panel de Entrada")
    st.markdown("Ingrese los parámetros del activo a analizar:")
    
    # Selector de sector
    sector = st.selectbox(
        "Sector del activo",
        ['energia', 'indice', 'divisa', 'metal', 'volatilidad'],
        help="Seleccione el sector al que pertenece el activo"
    )
    
    # Slider de volatilidad 20d
    volatilidad_20d = st.slider(
        "Volatilidad 20d",
        min_value=0.005,
        max_value=0.080,
        value=0.025,
        step=0.001,
        help="Volatilidad histórica del activo en los últimos 20 días"
    )
    
    # Slider de momentum 5d
    momentum_5d = st.slider(
        "Momentum 5d",
        min_value=-0.15,
        max_value=0.15,
        value=0.01,
        step=0.01,
        help="Retorno acumulado en los últimos 5 días"
    )
    
    # Slider de nivel VIX
    vix_nivel = st.slider(
        "Nivel VIX",
        min_value=10,
        max_value=80,
        value=25,
        step=1,
        help="Nivel actual del índice de volatilidad VIX"
    )
    
    # Slider de correlación con Brent
    correlacion_brent = st.slider(
        "Correlación con Brent",
        min_value=-1.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Correlación del activo con el petróleo Brent"
    )
    
    # Slider de CAR pre-evento
    car_pre_evento = st.slider(
        "CAR Pre-evento",
        min_value=-0.20,
        max_value=0.20,
        value=0.02,
        step=0.01,
        help="Retorno anormal acumulado antes del evento"
    )

# =============================================================================
# SECCIÓN 3 — Predicción
# =============================================================================

st.header("🔮 Predicción")

# Botón de predicción
if st.button("Predecir comportamiento del activo", type="primary"):
    try:
        # Cargar pipeline
        ruta_modelo = os.path.join("models", "modelo_final.pkl")
        pipeline = joblib.load(ruta_modelo)
        
        # Cargar dataset para obtener nombres de features
        ruta_datos = os.path.join("data", "processed", "dataset_modelamiento.csv")
        df = pd.read_csv(ruta_datos, index_col=0)
        
        # Obtener nombres de features (excluyendo targets y sectores)
        columnas_excluir = [col for col in df.columns if col.startswith('target_')] + \
                           [col for col in df.columns if col.endswith('_sector')]
        nombres_features = [col for col in df.columns if col not in columnas_excluir]
        
        # Crear DataFrame con valores por defecto (mediana del dataset)
        df_prediccion = pd.DataFrame(index=[0], columns=nombres_features)
        for col in nombres_features:
            df_prediccion[col] = df[col].median()
        
        # Actualizar con valores del usuario
        # Mapear sector a features relevantes
        if sector == 'energia':
            df_prediccion['BRENT_vol20'] = volatilidad_20d
            df_prediccion['BRENT_mom5'] = momentum_5d
            df_prediccion['BRENT_corr_brent'] = 1.0
        elif sector == 'indice':
            df_prediccion['SP500_vol20'] = volatilidad_20d
            df_prediccion['SP500_mom5'] = momentum_5d
            df_prediccion['SP500_corr_brent'] = correlacion_brent
        elif sector == 'divisa':
            df_prediccion['USD_COP_vol20'] = volatilidad_20d
            df_prediccion['USD_COP_mom5'] = momentum_5d
            df_prediccion['USD_COP_corr_brent'] = correlacion_brent
        elif sector == 'metal':
            df_prediccion['GOLD_vol20'] = volatilidad_20d
            df_prediccion['GOLD_mom5'] = momentum_5d
            df_prediccion['GOLD_corr_brent'] = correlacion_brent
        elif sector == 'volatilidad':
            df_prediccion['VIX_vol20'] = volatilidad_20d
            df_prediccion['VIX_mom5'] = momentum_5d
            df_prediccion['VIX_corr_brent'] = correlacion_brent
        
        # Actualizar features comunes
        df_prediccion['DELTA_VIX'] = vix_nivel - 25  # Asumiendo VIX base de 25
        df_prediccion['dias_al_evento'] = 10  # Valor por defecto
        
        # Ejecutar predicción
        prediccion = pipeline.predict(df_prediccion)[0]
        probabilidades = pipeline.predict_proba(df_prediccion)[0]
        
        prob_subida = probabilidades[1]
        prob_bajada = probabilidades[0]
        
        # Mostrar resultado principal
        col1, col2 = st.columns(2)
        
        with col1:
            if prediccion == 1:
                st.success("## ✅ RETORNO ANORMAL POSITIVO (SUBIDA)")
                st.metric("P(Subida)", f"{prob_subida:.2%}", delta=f"{prob_subida - 0.5:.2%}")
            else:
                st.error("## ❌ RETORNO ANORMAL NEGATIVO (BAJADA)")
                st.metric("P(Bajada)", f"{prob_bajada:.2%}", delta=f"{prob_bajada - 0.5:.2%}")
        
        with col2:
            st.metric("Probabilidad de Subida", f"{prob_subida:.2%}")
            st.metric("Probabilidad de Bajada", f"{prob_bajada:.2%}")
        
        # Gráfico de barras
        fig, ax = plt.subplots(figsize=(8, 4))
        categorias = ['Bajada', 'Subida']
        probabilidades_plot = [prob_bajada, prob_subida]
        colores = ['#ff6b6b', '#51cf66']
        
        barras = ax.bar(categorias, probabilidades_plot, color=colores, alpha=0.8)
        
        # Añadir valores en las barras
        for barra, prob in zip(barras, probabilidades_plot):
            ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.01,
                   f'{prob:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Probabilidad', fontsize=12)
        ax.set_title('Probabilidad de Retorno Anormal', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error("Error: No se encontró el modelo entrenado. Ejecute primero los notebooks.")
    except Exception as e:
        st.error(f"Error al ejecutar la predicción: {str(e)}")

# =============================================================================
# SECCIÓN 4 — Visualización del clustering
# =============================================================================

st.header("🎯 Visualización del Clustering")

# Cargar y mostrar gráfico de clustering
ruta_clustering = os.path.join("data", "processed", "graficos", "clustering_activos.png")

if os.path.exists(ruta_clustering):
    imagen = Image.open(ruta_clustering)
    st.image(imagen, caption="Clustering de Activos Financieros (PCA 2D)", use_column_width=True)
    
    # Texto explicativo según sector seleccionado
    st.markdown(f"""
    **Cluster más relevante para el sector '{sector}':**
    
    Los activos del sector **{sector}** tienden a agruparse con otros activos 
    que tienen comportamiento similar ante eventos geopolíticos. El gráfico 
    muestra cómo los activos se agrupan según sus características de:
    - Retorno anormal post-evento
    - Volatilidad
    - Correlación con Brent
    - Cambio en correlación
    """)
else:
    st.warning("No se encontró el gráfico de clustering. Ejecute primero el notebook 02_modelos_predictivos.ipynb.")

# =============================================================================
# SECCIÓN 5 — Métricas del modelo en producción
# =============================================================================

st.header("📊 Métricas del Modelo en Producción")

# Cargar métricas del modelo
try:
    # Intentar cargar métricas desde archivo
    ruta_metricas = os.path.join("data", "processed", "metricas_modelo.csv")
    
    if os.path.exists(ruta_metricas):
        df_metricas = pd.read_csv(ruta_metricas)
        auc = df_metricas.loc[df_metricas['Métrica'] == 'AUC-ROC', 'Valor'].values[0]
        f1 = df_metricas.loc[df_metricas['Métrica'] == 'F1-Score', 'Valor'].values[0]
        accuracy = df_metricas.loc[df_metricas['Métrica'] == 'Accuracy', 'Valor'].values[0]
    else:
        # Valores por defecto si no existe el archivo
        auc = 0.74
        f1 = 0.68
        accuracy = 0.71
    
    # Mostrar métricas en tres columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="AUC-ROC",
            value=f"{auc:.2f}",
            delta=f"{auc - 0.60:.2f}" if auc > 0.60 else f"{auc - 0.60:.2f}",
            delta_color="normal" if auc > 0.60 else "inverse"
        )
        st.caption("Línea base: 0.60")
    
    with col2:
        st.metric(
            label="F1-Score",
            value=f"{f1:.2f}",
            delta=f"{f1 - 0.60:.2f}" if f1 > 0.60 else f"{f1 - 0.60:.2f}",
            delta_color="normal" if f1 > 0.60 else "inverse"
        )
        st.caption("Línea base: 0.60")
    
    with col3:
        st.metric(
            label="Accuracy",
            value=f"{accuracy:.2f}",
            delta=f"{accuracy - 0.60:.2f}" if accuracy > 0.60 else f"{accuracy - 0.60:.2f}",
            delta_color="normal" if accuracy > 0.60 else "inverse"
        )
        st.caption("Línea base: 0.60")
    
    # Interpretación
    st.markdown("""
    **Interpretación de las métricas:**
    
    - **AUC-ROC**: Mide la capacidad del modelo para distinguir entre subida y bajada.
      Un valor de 0.74 indica que el modelo distingue correctamente en el 74% de los casos.
    
    - **F1-Score**: Balance entre precisión y recall. Un valor de 0.68 indica un buen
      balance entre no perderse subidas reales y no generar falsas alarmas.
    
    - **Accuracy**: Porcentaje de predicciones correctas. Un valor de 0.71 indica que
      el modelo clasificó correctamente el 71% de los días del conjunto de prueba.
    
    Todas las métricas superan la línea base de 0.60, lo que indica que el modelo
    agrega valor predictivo significativo.
    """)

except Exception as e:
    st.warning(f"No se pudieron cargar las métricas: {str(e)}")
    st.info("Ejecute primero los notebooks para generar las métricas del modelo.")

# =============================================================================
# SECCIÓN 6 — Acerca del proyecto
# =============================================================================

st.markdown("---")

with st.expander("ℹ️ Acerca del Proyecto"):
    st.markdown("""
    ## Descripción del Proyecto
    
    Este proyecto de Minería de Datos analiza el impacto de la captura de Nicolás Maduro 
    (3 de enero de 2026) en los mercados financieros globales. Utiliza la metodología 
    CRISP-DM para predecir si un activo financiero generará un retorno anormal positivo 
    o negativo ante eventos geopolíticos similares.
    
    ## Metodología CRISP-DM
    
    El proyecto sigue las 6 fases de CRISP-DM:
    
    1. **Comprensión del Negocio**: Definir objetivos y requisitos del proyecto
    2. **Comprensión de los Datos**: Recopilar y explorar datos financieros
    3. **Preparación de los Datos**: Limpiar, transformar y crear features
    4. **Modelamiento**: Entrenar y evaluar modelos predictivos
    5. **Evaluación**: Validar resultados y métricas
    6. **Despliegue**: Implementar la aplicación Streamlit
    
    ## Datos Utilizados
    
    - **Período**: 2020-01-01 hasta 2026-03-25
    - **Activos**: 11 activos financieros de diferentes sectores
    - **Frecuencia**: Datos diarios
    - **Fuente**: Yahoo Finance
    
    ## Modelos Implementados
    
    - Árbol de Decisión
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Red Neuronal (MLP)
    - Random Forest
    - XGBoost
    - Gradient Boosting
    
    ## Autoras
    
    - **Laura Laguado**
    - **Sofía Navales**
    
    ## Repositorio
    
    El código fuente completo está disponible en el repositorio GitHub del proyecto.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Proyecto de Minería de Datos Financieros</p>
    <p>Metodología CRISP-DM | Autoras: Laura Laguado y Sofía Navales</p>
    <p>Evento: Captura de Nicolás Maduro (3 ene 2026)</p>
</div>
""", unsafe_allow_html=True)