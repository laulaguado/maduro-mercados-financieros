# 📈 Minería de Datos: Impacto de la Captura de Maduro en Mercados Financieros Globales

## 📋 Descripción del Proyecto

Este proyecto de Minería de Datos analiza el impacto de la captura de Nicolás Maduro (3 de enero de 2026) en los mercados financieros globales. Utiliza la metodología CRISP-DM para predecir si un activo financiero generará un retorno anormal positivo (subida) o negativo (bajada) ante eventos geopolíticos similares.

### Pregunta de Investigación

¿Cómo impactó la captura de Nicolás Maduro en los mercados financieros globales, y es posible predecir el comportamiento de los activos ante eventos geopolíticos similares?

## 👥 Integrantes

- **Laura Laguado**
- **Sofía Navales**

## 📁 Estructura del Proyecto

```
proyecto_maduro_mercados/
│
├── src/
│   ├── data_collection.py       # Descarga y configuración de datos
│   ├── preprocessing.py         # Limpieza y transformaciones
│   ├── feature_engineering.py   # Creación de variables
│   ├── event_study.py           # Cálculo de AR y CAR
│   ├── models.py                # Entrenamiento de modelos
│   ├── evaluation.py            # Métricas y comparación
│   └── clustering.py            # Modelos de clustering
│
├── notebooks/
│   ├── 01_preparacion_datos.ipynb
│   ├── 02_modelos_predictivos.ipynb
│   └── 03_despliegue.ipynb
│
├── models/
│   └── modelo_final.pkl         # Pipeline serializado
│
├── data/
│   ├── raw/                     # Datos crudos descargados
│   └── processed/               # Datos limpios y transformados
│       └── graficos/            # Todas las visualizaciones generadas
│
├── app/
│   └── streamlit_app.py         # Interfaz gráfica de despliegue
│
├── requirements.txt             # Librerías con versiones fijas
└── README.md                    # Instrucciones completas del proyecto
```

## 🚀 Instrucciones de Instalación

### 1. Clonar el repositorio

```bash
git clone <url_del_repositorio>
cd proyecto_maduro_mercados
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 📊 Instrucciones para Ejecutar los Notebooks

Los notebooks deben ejecutarse en orden secuencial:

### Notebook 1: Preparación de Datos

```bash
jupyter notebook notebooks/01_preparacion_datos.ipynb
```

**Contenido:**
- Descarga de datos financieros desde Yahoo Finance
- Validación de calidad de datos
- Cálculo de retornos logarítmicos
- Detección y visualización de outliers
- Imputación de valores nulos
- Análisis de correlaciones
- Ingeniería de características
- Construcción del dataset final

### Notebook 2: Modelos Predictivos

```bash
jupyter notebook notebooks/02_modelos_predictivos.ipynb
```

**Contenido:**
- Event Study: cálculo de AR y CAR
- Creación de variable objetivo
- División de datos (70/30) y SMOTE
- Entrenamiento con validación cruzada (5-fold)
- ANOVA + Tukey para comparación de modelos
- Hiperparametrización de los 3 mejores modelos
- Evaluación final sobre el 30% de prueba
- Importancia de variables
- Clustering de activos financieros
- Construcción y guardado del pipeline final

### Notebook 3: Despliegue

```bash
jupyter notebook notebooks/03_despliegue.ipynb
```

**Contenido:**
- Carga y verificación del pipeline guardado
- Prueba del pipeline con datos nuevos de ejemplo
- Métricas finales del modelo en producción
- Lanzamiento de la app Streamlit
- Documentación de la interfaz de la app

## 🌐 Instrucciones para Lanzar la App Streamlit

### Opción 1: Desde el notebook

Ejecute la celda 4 del notebook `03_despliegue.ipynb`

### Opción 2: Desde la terminal

```bash
cd app
streamlit run streamlit_app.py
```

La app estará disponible en: http://localhost:8501

## 📈 Descripción de los Datos Utilizados

### Período de Análisis
- **Fecha de inicio:** 2020-01-01
- **Fecha de fin:** 2026-03-25
- **Fecha del evento:** 2026-01-03 (Captura de Maduro)

### Activos Financieros

| Nombre | Símbolo | Sector |
|--------|---------|--------|
| S&P 500 | ^GSPC | Índice |
| VIX | ^VIX | Volatilidad |
| Brent | BZ=F | Energía |
| WTI | CL=F | Energía |
| COLCAP | ^COLCAP | Índice |
| BOVESPA | ^BVSP | Índice |
| USD/COP | USDCOP=X | Divisa |
| Oro | GC=F | Metal |
| Cobre | HG=F | Metal |
| Exxon | XOM | Energía |
| Chevron | CVX | Energía |

### Fuente de Datos
- **Yahoo Finance**: Precios de cierre ajustados diarios

## 🎯 Resumen de Resultados Principales

### Event Study

Los retornos anormales (AR) y retornos anormales acumulados (CAR) muestran:

- **Petróleo (Brent, WTI)**: Reacción positiva significativa post-evento
- **Acciones petroleras (Exxon, Chevron)**: Comportamiento similar al petróleo
- **Índices bursátiles**: Reacción mixta según región
- **Metales (Oro, Cobre)**: Refugio seguro vs. demanda industrial
- **VIX**: Aumento de volatilidad post-evento

### Modelos Predictivos

Se entrenaron y evaluaron 7 modelos de clasificación:

1. Árbol de Decisión
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Red Neuronal (MLP)
5. Random Forest
6. XGBoost
7. Gradient Boosting

**Mejor modelo:** XGBoost con AUC-ROC de 0.74

### Clustering de Activos

Los activos se agruparon en clusters según su comportamiento:

- **Cluster 1**: Activos reactivos positivos (petróleo y acciones petroleras)
- **Cluster 2**: Activos de refugio seguro (oro, bonos)
- **Cluster 3**: Activos de alta volatilidad (VIX, índices emergentes)
- **Cluster 4**: Activos estables (índices desarrollados)

## 🔧 Tecnologías Utilizadas

- **Python 3.10+**
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib/Seaborn**: Visualización
- **Scikit-learn**: Machine Learning
- **XGBoost**: Gradient Boosting
- **Statsmodels**: Estadística
- **Streamlit**: Despliegue de aplicación
- **yfinance**: Descarga de datos financieros

## 📝 Notas Importantes

1. **Reproducibilidad**: Se utiliza `random_state=42` en todos los modelos
2. **Balanceo de clases**: SMOTE se aplica solo al 70% de entrenamiento
3. **Interpretación**: Todas las métricas se interpretan en español
4. **Visualizaciones**: Todos los gráficos incluyen línea vertical roja punteada marcando el evento

## 📚 Metodología CRISP-DM

El proyecto sigue las 6 fases de CRISP-DM:

1. **Comprensión del Negocio**: Definir objetivos y requisitos
2. **Comprensión de los Datos**: Recopilar y explorar datos
3. **Preparación de los Datos**: Limpiar, transformar y crear features
4. **Modelamiento**: Entrenar y evaluar modelos
5. **Evaluación**: Validar resultados y métricas
6. **Despliegue**: Implementar aplicación Streamlit

## 📧 Contacto

Para preguntas o comentarios sobre este proyecto, contacte a:
- Laura Laguado
- Sofía Navales

---

