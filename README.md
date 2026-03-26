# 📈 Impacto de la Captura de Nicolás Maduro en los Mercados Financieros Globales

> Proyecto final de Minería de Datos — Metodología CRISP-DM  
> Ingeniería Financiera & Ciencia de Datos | Medellín, Colombia | 2026

---

## 👩‍💻 Integrantes

| Nombre | Rol |
|---|---|
| Laura Laguado | Experto del Negocio / Modelamiento |
| Sofía Navales | Experto TI / Preparación de datos |

---

## 🎯 Pregunta de Investigación

> ¿Generó la captura de Nicolás Maduro el 3 de enero de 2026 retornos 
> anormales estadísticamente significativos en los mercados de petróleo, 
> renta variable latinoamericana, bonos soberanos venezolanos y divisas 
> de economías emergentes?

---

## 📌 Descripción del Proyecto

Este proyecto aplica la metodología **CRISP-DM** para analizar y predecir 
el comportamiento de 11 activos financieros globales ante el evento 
geopolítico de la captura de Nicolás Maduro (Operación Resolución 
Absoluta, 3 de enero de 2026).

Se desarrollan dos modelos principales:

- **Clasificación**: predecir si un activo generará un retorno anormal 
  positivo (subida) o negativo (bajada) ante eventos geopolíticos extremos.
- **Clustering**: agrupar activos financieros según su comportamiento 
  pre y post evento para construir estrategias de diversificación.

---

## 🗂️ Estructura del Repositorio
```
maduro-mercados-financieros/
│
├── src/                         # Módulos Python del proyecto
│   ├── data_collection.py       # Descarga de datos desde Yahoo Finance
│   ├── preprocessing.py         # Limpieza y transformaciones
│   ├── feature_engineering.py   # Ingeniería de características
│   ├── event_study.py           # Cálculo de retornos anormales (AR/CAR)
│   ├── models.py                # Entrenamiento y selección de modelos
│   ├── evaluation.py            # Métricas y visualizaciones
│   └── clustering.py            # Clustering de activos
│
├── notebooks/
│   ├── 01_preparacion_datos.ipynb      # CRISP-DM: Fases 1-3
│   ├── 02_modelos_predictivos.ipynb    # CRISP-DM: Fase 4-5
│   └── 03_despliegue.ipynb            # CRISP-DM: Fase 5 (despliegue)
│
├── models/
│   └── modelo_final.pkl         # Pipeline serializado (generado al correr nb2)
│
├── data/
│   ├── raw/                     # Datos crudos de Yahoo Finance
│   └── processed/               # Datos limpios y transformados
│       └── graficos/            # Visualizaciones generadas
│
├── app/
│   └── streamlit_app.py         # App de predicción en tiempo real
│
├── requirements.txt
└── README.md
```

---

## 📊 Datos Utilizados

| Activo | Ticker Yahoo Finance | Clase |
|---|---|---|
| S&P 500 | ^GSPC | Índice |
| VIX | ^VIX | Volatilidad |
| Petróleo Brent | BZ=F | Energía |
| Petróleo WTI | CL=F | Energía |
| COLCAP Colombia | ^COLCAP | Índice |
| Bovespa Brasil | ^BVSP | Índice |
| USD/COP | USDCOP=X | Divisa |
| Oro | GC=F | Metal |
| Cobre | HG=F | Metal |
| ExxonMobil | XOM | Energía |
| Chevron | CVX | Energía |

- **Período**: Enero 2020 – Marzo 2026
- **Frecuencia**: Diaria (días hábiles bursátiles)
- **Registros**: ~17.000 filas en dataset integrado
- **Fuente**: Yahoo Finance vía librería `yfinance`

---

## 🤖 Modelos Implementados

### Clasificación (variable objetivo: retorno anormal positivo/negativo)

| # | Modelo | Tipo |
|---|---|---|
| 1 | Árbol de Decisión | Supervisado |
| 2 | K-Nearest Neighbors | Supervisado |
| 3 | Support Vector Machine | Supervisado |
| 4 | Red Neuronal (MLP) | Supervisado |
| 5 | Random Forest | Ensamble |
| 6 | XGBoost | Ensamble |
| 7 | Gradient Boosting | Ensamble |

### Clustering (agrupación de activos)

| Algoritmo | Métrica de evaluación |
|---|---|
| K-Means | Silhouette + Método del codo |
| Jerárquico (Ward) | Dendrograma |
| DBSCAN | Silhouette + Davies-Bouldin |

---

## ⚙️ Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/maduro-mercados-financieros.git
cd maduro-mercados-financieros
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Mac/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar los notebooks en orden
```bash
# Abrir Jupyter
jupyter notebook

# Ejecutar en este orden:
# 1. notebooks/01_preparacion_datos.ipynb
# 2. notebooks/02_modelos_predictivos.ipynb
# 3. notebooks/03_despliegue.ipynb
```

### 5. Lanzar la app Streamlit
```bash
cd app
streamlit run streamlit_app.py
```
La app estará disponible en: http://localhost:8501

---

## 📐 Metodología

Este proyecto sigue las 5 fases de **CRISP-DM**:

| Fase | Descripción | Notebook |
|---|---|---|
| 1. Entendimiento del negocio | Contexto geopolítico y financiero | Documentación |
| 2. Entendimiento de los datos | Exploración y calidad de datos | nb01 |
| 3. Preparación de datos | Limpieza, features, balanceo | nb01 |
| 4. Modelamiento | 7 modelos + ANOVA + Tukey + hiperparámetros | nb02 |
| 5. Despliegue | Pipeline + Streamlit | nb03 |

### Medidas matemáticas clave

- **Retorno logarítmico**: `r_t = ln(P_t / P_{t-1})`
- **Retorno Anormal**: `AR_t = R_activo_t − (α + β × R_mercado_t)`
- **Retorno Anormal Acumulado**: `CAR = Σ AR_t` en la ventana del evento
- **Variable objetivo**: `target = 1 si AR > 0, else 0`
- **Línea base**: P = 60% | **Meta**: AUC-ROC > 0.70

---

## 📋 Entregables

- [x] Documentación CRISP-DM (Word)
- [x] Notebook 1: Preparación de datos (con pandas profiling)
- [x] Notebook 2: Modelos predictivos y clustering
- [x] Notebook 3: Despliegue
- [x] App Streamlit
- [x] Repositorio GitHub

---

## 📚 Referencias

- Fama et al. (1969). *The Adjustment of Stock Prices to New Information*
- MacKinlay (1997). *Event Studies in Economics and Finance*
- Bollerslev (1986). *Generalized Autoregressive Conditional Heteroskedasticity*
- Datos: Yahoo Finance API — yfinance

---

*Proyecto académico — Minería de Datos | 2026*
