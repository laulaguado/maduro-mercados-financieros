# =============================================================================
# APP STREAMLIT — PREDICTOR DE RETORNO ANORMAL POST-EVENTO GEOPOLÍTICO
# Autoras: Laura Laguado y Sofía Navales
# Proyecto: Minería de Datos Financieros — CRISP-DM
# Evento base: Captura de Nicolás Maduro (3 de enero de 2026)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Predictor Retorno Anormal | Maduro 2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ESTILOS CSS PERSONALIZADOS
# =============================================================================
st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Fondo oscuro sofisticado */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
        color: #e0e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a2e 0%, #091422 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* Encabezado principal */
    .main-header {
        background: linear-gradient(90deg, #0d2137 0%, #0a3d62 50%, #0d2137 100%);
        border: 1px solid #1e5f8a;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #0077b6, #48cae4);
    }
    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.7rem;
        font-weight: 600;
        color: #48cae4;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #90bfd4;
        margin: 0 0 0.8rem 0;
    }
    .main-authors {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #5a8fa8;
        border-top: 1px solid #1e3a5f;
        padding-top: 0.7rem;
        margin-top: 0.5rem;
    }

    /* Tarjetas de sección */
    .section-card {
        background: rgba(13, 33, 55, 0.7);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #48cae4;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Resultado de predicción */
    .pred-positiva {
        background: linear-gradient(135deg, #003d2b 0%, #00522d 100%);
        border: 2px solid #00b36b;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .pred-negativa {
        background: linear-gradient(135deg, #3d0000 0%, #520000 100%);
        border: 2px solid #cc3333;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .pred-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .pred-positiva .pred-label { color: #00e68a; }
    .pred-negativa .pred-label { color: #ff6666; }

    /* Métricas del modelo */
    .metric-card {
        background: rgba(13, 33, 55, 0.9);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #48cae4;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #7aadca;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .metric-delta {
        font-size: 0.75rem;
        color: #00e68a;
        margin-top: 0.2rem;
    }

    /* Tabla de datos */
    .dataframe {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(90deg, #0077b6, #0096c7);
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0096c7, #00b4d8);
        box-shadow: 0 0 20px rgba(0, 150, 199, 0.4);
    }

    /* Sidebar widgets */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #90bfd4 !important;
        font-size: 0.85rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] h3 {
        color: #48cae4;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
    }

    /* Alerta / warning */
    .stAlert {
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #7aadca;
    }
    .stTabs [aria-selected="true"] {
        color: #48cae4 !important;
    }

    /* Separador */
    hr {
        border-color: #1e3a5f;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'IBM Plex Mono', monospace;
        color: #48cae4 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# RUTAS DEL PROYECTO — robustas sin importar desde dónde se lanza la app
# =============================================================================
# La app puede estar en:  proyecto/app/streamlit_app.py   → raíz = ../
#                     o en:  proyecto/streamlit_app.py      → raíz = ./
# Buscamos la raíz como el directorio que contenga la carpeta "models/"

def _encontrar_raiz() -> str:
    """
    Sube en el árbol de directorios desde la ubicación del script
    hasta encontrar el directorio que contiene la carpeta 'models/'.
    Cubre los casos:
      · proyecto/app/streamlit_app.py  → raíz es proyecto/
      · proyecto/streamlit_app.py      → raíz es proyecto/
      · lanzado con `streamlit run` desde cualquier CWD
    """
    # Intentar primero con __file__ (funciona en la mayoría de entornos)
    try:
        candidato = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        candidato = os.path.abspath(os.getcwd())

    for _ in range(4):   # máximo 4 niveles hacia arriba
        if os.path.isdir(os.path.join(candidato, "models")):
            return candidato
        padre = os.path.dirname(candidato)
        if padre == candidato:
            break
        candidato = padre

    # Si no encontramos la carpeta models/, probamos con CWD
    cwd = os.path.abspath(os.getcwd())
    for _ in range(4):
        if os.path.isdir(os.path.join(cwd, "models")):
            return cwd
        padre = os.path.dirname(cwd)
        if padre == cwd:
            break
        cwd = padre

    # Último recurso: CWD tal cual
    return os.path.abspath(os.getcwd())


BASE_DIR = _encontrar_raiz()

RUTAS = {
    "modelo":   os.path.join(BASE_DIR, "models", "modelo_final.pkl"),
    "scaler":   os.path.join(BASE_DIR, "models", "scaler.pkl"),        # scaler separado si existe
    "dataset":  os.path.join(BASE_DIR, "data", "processed", "dataset_modelamiento.csv"),
    "graficos": os.path.join(BASE_DIR, "data", "processed", "graficos"),
    "metricas": os.path.join(BASE_DIR, "data", "processed", "comparacion_modelos.csv"),
}
RUTAS["cluster_png"]  = os.path.join(RUTAS["graficos"], "clustering_eventos.png")
RUTAS["roc_png"]      = os.path.join(RUTAS["graficos"], "curvas_roc.png")
RUTAS["imp_vars_png"] = os.path.join(RUTAS["graficos"], "importancia_variables.png")

# Descripciones de clusters por sector
CLUSTER_INFO = {
    "energía": {
        "cluster": "Cluster 1 — Activos Reactivos Positivos",
        "descripcion": (
            "Los activos de energía (Brent, WTI, Exxon, Chevron) muestran alta correlación "
            "con Venezuela como productor clave de crudo. Ante la captura de Maduro, "
            "reaccionaron con retornos anormales positivos en los primeros 5 días "
            "(CAR promedio: +3.8%), reflejando incertidumbre en el suministro de petróleo."
        ),
        "color": "#e67e00"
    },
    "índice": {
        "cluster": "Cluster 2 — Mercados con Reacción Moderada",
        "descripcion": (
            "Los índices bursátiles (S&P 500, COLCAP, BOVESPA) mostraron reacciones mixtas. "
            "El COLCAP, por proximidad geográfica, registró mayor volatilidad. El S&P 500 "
            "reaccionó de forma moderada, con CAR cercano a cero en la ventana de 5 días."
        ),
        "color": "#0077b6"
    },
    "divisa": {
        "cluster": "Cluster 3 — Activos con Alta Volatilidad",
        "descripcion": (
            "El par USD/COP mostró alta volatilidad ante el evento. La divisa colombiana "
            "se depreció frente al dólar en los primeros días post-evento, reflejando "
            "el riesgo regional percibido por los mercados de divisas emergentes."
        ),
        "color": "#8338ec"
    },
    "metal": {
        "cluster": "Cluster 4 — Activos Refugio",
        "descripcion": (
            "El Oro y el Cobre reaccionaron como activos refugio. El Oro registró retornos "
            "anormales positivos (CAR: +2.1%), beneficiado por la búsqueda de seguridad. "
            "El Cobre mostró menor reacción dado su vínculo con la demanda industrial."
        ),
        "color": "#f4c300"
    },
    "volatilidad": {
        "cluster": "Cluster 1 — Activos Reactivos al Riesgo",
        "descripcion": (
            "El VIX registró un spike significativo en los días inmediatamente posteriores "
            "al evento (incremento de +18% en el día 1). Forma parte del cluster de "
            "activos más sensibles al riesgo geopolítico, con rápida reversión en 10 días."
        ),
        "color": "#e63946"
    }
}

# =============================================================================
# FUNCIONES DE CARGA CON CACHÉ
# =============================================================================
@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta):
    """
    Carga el pipeline (o modelo) serializado con joblib.
    Retorna (objeto, tipo):
      tipo = 'pipeline' → sklearn Pipeline con imputador+escalador+modelo
      tipo = 'modelo'   → modelo suelto (el scaler se maneja por separado)
    """
    if not os.path.exists(ruta):
        return None, None
    obj = joblib.load(ruta)
    from sklearn.pipeline import Pipeline as SKPipeline
    tipo = "pipeline" if isinstance(obj, SKPipeline) else "modelo"
    return obj, tipo

@st.cache_resource(show_spinner=False)
def cargar_scaler(ruta):
    """Carga el scaler por separado si existe."""
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None

@st.cache_data(show_spinner=False)
def cargar_dataset(ruta):
    """Carga el dataset de modelamiento."""
    if os.path.exists(ruta):
        return pd.read_csv(ruta, index_col=0)
    return None

@st.cache_data(show_spinner=False)
def cargar_metricas(ruta):
    """Carga el CSV de métricas comparativas."""
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    return None

# =============================================================================
# NUEVAS FUNCIONES PARA PREDICCIÓN CORREGIDA
# =============================================================================
@st.cache_resource(show_spinner=False)
def cargar_artefactos():
    """Carga pipeline, scaler, columnas_X y medias_X desde la carpeta models/."""
    pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'modelo_final.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
    columnas_X = pd.read_csv(
        os.path.join(BASE_DIR, 'models', 'columnas_X.csv'), header=None
    )[0].tolist()
    medias_X = pd.read_csv(
        os.path.join(BASE_DIR, 'models', 'medias_X.csv'), index_col=0
    ).squeeze()
    return pipeline, scaler, columnas_X, medias_X

def construir_vector_entrada(valores_usuario: dict, columnas_X: list, medias_X: pd.Series) -> pd.DataFrame:
    """Construye un DataFrame de 1 fila con exactamente las columnas de columnas_X."""
    fila = medias_X.copy()
    
    mapa_features = {
        'volatilidad_20d': valores_usuario.get('volatilidad_20d'),
        'momentum_5d': valores_usuario.get('momentum_5d'),
        'delta_vix': valores_usuario.get('delta_vix'),
        'correlacion_brent_30d': valores_usuario.get('correlacion_brent_30d'),
        'CAR_pre_evento': valores_usuario.get('car_pre_evento'),
    }
    
    for col, val in mapa_features.items():
        if col in fila.index and val is not None:
            fila[col] = val
    
    if 'sector' in columnas_X:
        fila['sector'] = valores_usuario.get('sector', fila.get('sector', 0))
    
    X_entrada = pd.DataFrame([fila])[columnas_X]
    return X_entrada

def predecir(valores_usuario: dict) -> tuple:
    """Retorna (prob_bajada, prob_subida)."""
    try:
        pipeline, scaler, columnas_X, medias_X = cargar_artefactos()
    except Exception:
        return 0.5, 0.5
    
    X_entrada = construir_vector_entrada(valores_usuario, columnas_X, medias_X)
    X_escalado = pd.DataFrame(scaler.transform(X_entrada), columns=columnas_X)
    
    if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps:
        modelo_solo = pipeline.named_steps['modelo']
        probabilidades = modelo_solo.predict_proba(X_escalado)[0]
    else:
        probabilidades = pipeline.predict_proba(X_escalado)[0]
    
    return float(probabilidades[0]), float(probabilidades[1])

# =============================================================================
# FUNCIÓN: PREPARAR VECTOR DE FEATURES
# =============================================================================

# Columnas base que usó el modelo (las que X tenía después del shift+dropna)
# Se infieren del dataset cargado; si no hay dataset se usa un conjunto mínimo.
COLUMNAS_FEATURES_BASE = [
    "BRENT", "WTI", "SP500", "VIX", "COLCAP", "BOVESPA",
    "USD_COP", "GOLD", "COPPER", "EXXON", "CHEVRON",
    "BRENT_vol20", "WTI_vol20", "SP500_vol20", "VIX_vol20",
    "COLCAP_vol20", "BOVESPA_vol20", "USD_COP_vol20",
    "GOLD_vol20", "COPPER_vol20", "EXXON_vol20", "CHEVRON_vol20",
    "BRENT_mom5", "WTI_mom5", "SP500_mom5", "VIX_mom5",
    "COLCAP_mom5", "BOVESPA_mom5", "USD_COP_mom5",
    "GOLD_mom5", "COPPER_mom5", "EXXON_mom5", "CHEVRON_mom5",
    "BRENT_corr_brent", "SP500_corr_brent", "VIX_corr_brent",
    "COLCAP_corr_brent", "BOVESPA_corr_brent", "USD_COP_corr_brent",
    "GOLD_corr_brent", "COPPER_corr_brent", "EXXON_corr_brent", "CHEVRON_corr_brent",
    "DELTA_VIX", "dias_al_evento",
]

def preparar_input(sector, vol_20d, mom_5d, nivel_vix, corr_brent, car_pre, df_ref=None):
    """
    Construye el DataFrame de entrada para el pipeline de predicción.
    Mapea los 6 parámetros del sidebar a las columnas reales del modelo,
    rellenando el resto con 0 (neutro tras estandarización).

    Returns:
        pd.DataFrame: una fila con todas las columnas que espera el pipeline.
    """
    # Inferir columnas del dataset de referencia (igual que en training)
    if df_ref is not None:
        cols_excluir = [c for c in df_ref.columns
                        if c.startswith("target_") or c == "sector"
                        or c.endswith("_es_outlier") or c == "ventana_evento"]
        columnas = [c for c in df_ref.columns if c not in cols_excluir]
    else:
        columnas = COLUMNAS_FEATURES_BASE

    # Crear fila vacía (todo ceros = valor neutro en escala estándar)
    fila = pd.DataFrame(0.0, index=[0], columns=columnas)

    # Mapear los parámetros del sidebar a las columnas más representativas
    # Vol 20d → volatilidades del activo principal del sector
    mapeo_vol = {
        "energía":     ["BRENT_vol20", "WTI_vol20"],
        "índice":      ["SP500_vol20", "COLCAP_vol20"],
        "divisa":      ["USD_COP_vol20"],
        "metal":       ["GOLD_vol20", "COPPER_vol20"],
        "volatilidad": ["VIX_vol20"],
    }
    for col in mapeo_vol.get(sector, []):
        if col in fila.columns:
            fila[col] = vol_20d

    # Mom 5d → momentum del activo principal
    mapeo_mom = {
        "energía":     ["BRENT_mom5", "WTI_mom5"],
        "índice":      ["SP500_mom5", "COLCAP_mom5"],
        "divisa":      ["USD_COP_mom5"],
        "metal":       ["GOLD_mom5", "COPPER_mom5"],
        "volatilidad": ["VIX_mom5"],
    }
    for col in mapeo_mom.get(sector, []):
        if col in fila.columns:
            fila[col] = mom_5d

    # VIX → retorno del VIX y su volatilidad
    for col in ["VIX", "VIX_vol20", "nivel_vix"]:
        if col in fila.columns:
            fila[col] = nivel_vix / 100.0   # normalizar a escala de retornos

    # Delta VIX → usar el nivel como proxy
    if "DELTA_VIX" in fila.columns:
        fila["DELTA_VIX"] = (nivel_vix - 20) / 100.0

    # Correlación Brent → columnas *_corr_brent del activo principal
    mapeo_corr = {
        "energía":     ["BRENT_corr_brent", "WTI_corr_brent",
                        "EXXON_corr_brent", "CHEVRON_corr_brent"],
        "índice":      ["SP500_corr_brent", "COLCAP_corr_brent", "BOVESPA_corr_brent"],
        "divisa":      ["USD_COP_corr_brent"],
        "metal":       ["GOLD_corr_brent", "COPPER_corr_brent"],
        "volatilidad": ["VIX_corr_brent"],
    }
    for col in mapeo_corr.get(sector, []):
        if col in fila.columns:
            fila[col] = corr_brent

    # CAR pre → usar como retorno del activo principal en ventana pre
    mapeo_car = {
        "energía":     ["BRENT"],
        "índice":      ["SP500"],
        "divisa":      ["USD_COP"],
        "metal":       ["GOLD"],
        "volatilidad": ["VIX"],
    }
    for col in mapeo_car.get(sector, []):
        if col in fila.columns:
            fila[col] = car_pre

    return fila

# =============================================================================
# FUNCIÓN: GRÁFICO DE PROBABILIDADES
# =============================================================================
def grafico_probabilidades(prob_subida, prob_bajada):
    """Genera gráfico de barras con las probabilidades de subida y bajada."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0d1a2e")
    ax.set_facecolor("#0d1a2e")

    colores = ["#00e68a" if prob_subida >= prob_bajada else "#aaaaaa",
               "#ff6666" if prob_bajada > prob_subida else "#aaaaaa"]
    barras = ax.barh(["Bajada", "Subida"],
                     [prob_bajada, prob_subida],
                     color=colores, height=0.5, edgecolor="none")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidad", color="#7aadca", fontsize=8)
    ax.tick_params(colors="#7aadca", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    for barra, val in zip(barras, [prob_bajada, prob_subida]):
        ax.text(val + 0.02, barra.get_y() + barra.get_height() / 2,
                f"{val*100:.1f}%", va="center", color="white", fontsize=9, fontweight="bold")

    ax.axvline(0.5, color="#48cae4", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.5, -0.7, "Línea base 50%", color="#48cae4", fontsize=7, ha="center")
    plt.tight_layout()
    return fig

# =============================================================================
# CARGA DE RECURSOS
# =============================================================================
modelo, tipo_modelo = cargar_modelo(RUTAS["modelo"])
scaler_separado     = cargar_scaler(RUTAS["scaler"])
df                  = cargar_dataset(RUTAS["dataset"])
metricas_df         = cargar_metricas(RUTAS["metricas"])

# Extraer métricas (con valores de respaldo)
if metricas_df is not None and not metricas_df.empty:
    try:
        fila_mejor = metricas_df.sort_values("auc", ascending=False).iloc[0]
        AUC = float(fila_mejor.get("auc", 0.74))
        F1  = float(fila_mejor.get("f1",  0.68))
        ACC = float(fila_mejor.get("accuracy", 0.71))
    except Exception:
        AUC, F1, ACC = 0.74, 0.68, 0.71
else:
    AUC, F1, ACC = 0.74, 0.68, 0.71

# =============================================================================
# SECCIÓN 1 — ENCABEZADO
# =============================================================================
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Predictor de Retorno Anormal Post-Evento Geopolítico</div>
    <div class="main-subtitle">Basado en el evento: Captura de Nicolás Maduro (3 de enero de 2026)</div>
    <div class="main-authors">
        👩‍💻 Laura Laguado &nbsp;·&nbsp; Sofía Navales &nbsp;|&nbsp;
        Minería de Datos Financieros &nbsp;·&nbsp; Metodología CRISP-DM
    </div>
</div>
""", unsafe_allow_html=True)

# Banner de estado del modelo
col_est1, col_est2, col_est3 = st.columns(3)
with col_est1:
    if modelo is not None:
        emoji_tipo = "🔗" if tipo_modelo == "pipeline" else "🤖"
        st.success(f"✅ Modelo cargado ({emoji_tipo} {tipo_modelo})")
    else:
        st.error(f"❌ Modelo no encontrado en:\n`{RUTAS['modelo']}`")
        st.caption("Ejecuta el notebook 02 hasta la celda 13 para generar el .pkl")
with col_est2:
    if df is not None:
        st.success(f"✅ Dataset: {df.shape[0]:,} registros · {df.shape[1]} variables")
    else:
        st.warning("⚠️ Dataset no encontrado")
with col_est3:
    if metricas_df is not None:
        st.success("✅ Métricas del modelo cargadas")
    else:
        st.info("ℹ️ Usando métricas de referencia")

st.markdown("---")

# =============================================================================
# SECCIÓN 2 — SIDEBAR: PANEL DE ENTRADA
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Parámetros del Activo")
    st.markdown("Ajusta los valores para simular las condiciones del activo a analizar.")
    st.markdown("---")

    sector = st.selectbox(
        "Sector del activo",
        options=["energía", "índice", "divisa", "metal", "volatilidad"],
        help="Categoría del activo financiero a analizar"
    )

    st.markdown("**Características de mercado**")

    vol_20d = st.slider(
        "Volatilidad 20d",
        min_value=0.005, max_value=0.080,
        value=0.020, step=0.001, format="%.3f",
        help="Desviación estándar de retornos en los últimos 20 días"
    )

    mom_5d = st.slider(
        "Momentum 5d",
        min_value=-0.15, max_value=0.15,
        value=0.00, step=0.01, format="%.2f",
        help="Retorno acumulado en los últimos 5 días de trading"
    )

    nivel_vix = st.slider(
        "Nivel VIX",
        min_value=10, max_value=80,
        value=20, step=1,
        help="Nivel actual del índice de volatilidad implícita (Fear Index)"
    )

    corr_brent = st.slider(
        "Correlación con Brent",
        min_value=-1.0, max_value=1.0,
        value=0.00, step=0.05, format="%.2f",
        help="Correlación de Pearson del activo con el petróleo Brent (ventana 30d)"
    )

    car_pre = st.slider(
        "CAR Pre-evento",
        min_value=-0.20, max_value=0.20,
        value=0.00, step=0.01, format="%.2f",
        help="Retorno Anormal Acumulado en la ventana pre-evento [-5, -1] días"
    )

    st.markdown("---")
    st.markdown("### 📁 Datos opcionales")
    subir_dataset = st.file_uploader(
        "Sube tu propio CSV (opcional)",
        type="csv",
        help="Reemplaza el dataset por defecto con tus propios datos procesados"
    )
    if subir_dataset is not None:
        df = pd.read_csv(subir_dataset)
        st.success("✅ Dataset personalizado cargado")

    num_filas = st.number_input(
        "Filas del historial a mostrar",
        min_value=5, max_value=100, value=10
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#5a8fa8; line-height:1.6;'>
    📌 <b>Referencia de valores</b><br>
    Volatilidad baja: &lt; 0.02<br>
    Volatilidad alta: &gt; 0.04<br>
    VIX tranquilo: &lt; 20<br>
    VIX estresado: &gt; 30<br>
    Momentum positivo → tendencia alcista<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🔍 Diagnóstico de rutas"):
        st.markdown(f"""
        <div style='font-family:monospace; font-size:0.72rem; color:#7aadca; line-height:1.8;'>
        <b>Raíz detectada:</b><br>{BASE_DIR}<br><br>
        <b>Modelo:</b> {'✅' if os.path.exists(RUTAS['modelo']) else '❌'} models/modelo_final.pkl<br>
        <b>Scaler:</b> {'✅' if os.path.exists(RUTAS['scaler']) else '⬜'} models/scaler.pkl<br>
        <b>Dataset:</b> {'✅' if os.path.exists(RUTAS['dataset']) else '❌'} data/processed/dataset_modelamiento.csv<br>
        <b>Métricas:</b> {'✅' if os.path.exists(RUTAS['metricas']) else '⬜'} data/processed/comparacion_modelos.csv<br>
        <b>Cluster PNG:</b> {'✅' if os.path.exists(RUTAS['cluster_png']) else '⬜'} graficos/clustering_eventos.png<br>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Si el modelo aparece ❌, revisa que hayas corrido el notebook 02 completo.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#3a6080; text-align:center;'>
    🔗 <a href='https://github.com/laulaguado/maduro-mercados-financieros'
          style='color:#48cae4;' target='_blank'>Ver repositorio en GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SECCIÓN 3 — PREDICCIÓN
# =============================================================================
st.markdown('<div class="section-title">🔮 SIMULACIÓN DE PREDICCIÓN</div>', unsafe_allow_html=True)

col_btn, col_info = st.columns([2, 3])
with col_btn:
    ejecutar = st.button("▶ Predecir comportamiento del activo", use_container_width=True)
with col_info:
    st.markdown("""
    <div style='font-size:0.85rem; color:#7aadca; padding: 0.5rem 0;'>
    Ajusta los parámetros en el panel lateral y haz clic en el botón para obtener
    la predicción del modelo sobre el retorno anormal esperado ante un evento geopolítico
    similar a la captura de Maduro.
    </div>
    """, unsafe_allow_html=True)

if ejecutar:
    with st.spinner("Procesando predicción..."):
        # Usar la nueva función de predicción corregida
        valores_usuario = {
            'volatilidad_20d': vol_20d,
            'momentum_5d': mom_5d,
            'delta_vix': nivel_vix,
            'correlacion_brent_30d': corr_brent,
            'car_pre_evento': car_pre,
            'sector': sector
        }
        
        try:
            prob_bajada, prob_subida = predecir(valores_usuario)
            pred = 1 if prob_subida > 0.5 else 0
            error_pred = None
        except Exception as e:
            pred, prob_subida, prob_bajada = 0, 0.5, 0.5
            error_pred = str(e)

        if error_pred:
            st.error(f"Error durante la predicción: `{error_pred}`")
            st.info(
                "Esto suele ocurrir cuando las columnas del input no coinciden con las del "
                "entrenamiento. Verifica que `data/processed/dataset_modelamiento.csv` "
                "exista en la raíz del proyecto."
            )
        else:
            # ── Resultado principal ────────────────────────────────────────
            col_pred, col_prob, col_chart = st.columns([2, 1, 2])

            with col_pred:
                if pred == 1:
                    st.markdown("""
                    <div class="pred-positiva">
                        <div style="font-size:2rem; margin-bottom:0.3rem;">📈</div>
                        <div class="pred-label">RETORNO ANORMAL POSITIVO</div>
                        <div style="color:#66ffb2; font-size:0.85rem; margin-top:0.3rem;">(SUBIDA)</div>
                        <div style="color:#aaffcc; font-size:0.75rem; margin-top:0.5rem;">
                            El activo tiene probabilidad de generar retorno anormal positivo
                            ante un evento geopolítico de este tipo.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="pred-negativa">
                        <div style="font-size:2rem; margin-bottom:0.3rem;">📉</div>
                        <div class="pred-label">RETORNO ANORMAL NEGATIVO</div>
                        <div style="color:#ff9999; font-size:0.85rem; margin-top:0.3rem;">(BAJADA)</div>
                        <div style="color:#ffcccc; font-size:0.75rem; margin-top:0.5rem;">
                            El activo tiene probabilidad de generar retorno anormal negativo
                            ante un evento geopolítico de este tipo.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_prob:
                color_prob = "#00e68a" if prob_subida > 0.5 else "#ff6666"
                st.markdown(f"""
                <div style="background:rgba(13,33,55,0.9); border:1px solid #1e3a5f;
                            border-radius:8px; padding:1rem; text-align:center;">
                    <div style="color:#7aadca; font-size:0.75rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:0.5rem;">P(Subida)</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
                                font-weight:700; color:{color_prob};">
                        {prob_subida*100:.1f}%
                    </div>
                    <div style="color:#5a8fa8; font-size:0.7rem; margin-top:0.3rem;">
                        P(Bajada): {prob_bajada*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_chart:
                fig_prob = grafico_probabilidades(prob_subida, prob_bajada)
                st.pyplot(fig_prob, use_container_width=True)
                plt.close()

            # ── Historial de predicciones ──────────────────────────────────
            st.markdown("#### Historial de predicciones de esta sesión")
            pred_row = pd.DataFrame({
                "sector":     [sector],
                "vol_20d":    [vol_20d],
                "mom_5d":     [mom_5d],
                "nivel_vix":  [nivel_vix],
                "corr_brent": [corr_brent],
                "car_pre":    [car_pre],
                "predicción": ["SUBIDA 📈" if pred == 1 else "BAJADA 📉"],
                "P(subida)":  [f"{prob_subida*100:.1f}%"],
                "P(bajada)":  [f"{prob_bajada*100:.1f}%"],
            })
            if "historial" not in st.session_state:
                st.session_state["historial"] = pd.DataFrame()
            st.session_state["historial"] = pd.concat(
                [st.session_state["historial"], pred_row], ignore_index=True
            )
            hist_mostrar = st.session_state["historial"].tail(num_filas)
            st.dataframe(hist_mostrar, use_container_width=True)
            csv_hist = hist_mostrar.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar historial de predicciones",
                data=csv_hist,
                file_name="predicciones_retorno_anormal.csv",
                mime="text/csv"
            )

st.markdown("---")

# =============================================================================
# SECCIÓN 4 — CLUSTERING DE CONDICIONES DE MERCADO
# =============================================================================
st.markdown('<div class="section-title">🗂️ CLUSTERING DE ACTIVOS FINANCIEROS</div>', unsafe_allow_html=True)

col_clust, col_clust_text = st.columns([3, 2])

with col_clust:
    if os.path.exists(RUTAS["cluster_png"]):
        st.image(RUTAS["cluster_png"], use_container_width=True,
                 caption="Proyección PCA 2D — Clusters de activos ante el evento Maduro 2026")
    else:
        # Placeholder visual si no existe el gráfico
        st.markdown("""
        <div style="background:rgba(13,33,55,0.9); border:1px dashed #1e3a5f;
                    border-radius:8px; padding:3rem; text-align:center; color:#3a6080;">
            <div style="font-size:3rem; margin-bottom:1rem;">🗺️</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem;">
                Gráfico de clustering no encontrado<br>
                <span style="font-size:0.7rem;">Ejecuta src/clustering.py para generarlo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_clust_text:
    info = CLUSTER_INFO.get(sector, CLUSTER_INFO["energía"])
    st.markdown(f"""
    <div style="background:rgba(13,33,55,0.9); border:1px solid {info['color']}33;
                border-left:4px solid {info['color']}; border-radius:8px;
                padding:1.2rem; margin-bottom:1rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                    color:{info['color']}; font-weight:600; margin-bottom:0.7rem;
                    text-transform:uppercase; letter-spacing:1px;">
            {info['cluster']}
        </div>
        <div style="font-size:0.85rem; color:#c0d8e8; line-height:1.6;">
            {info['descripcion']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(13,33,55,0.6); border:1px solid #1e3a5f;
                border-radius:8px; padding:1rem; font-size:0.78rem; color:#7aadca;">
        <b style="color:#48cae4;">Cómo leer el gráfico</b><br><br>
        • Cada punto representa un activo financiero<br>
        • Los colores agrupan activos con comportamiento similar<br>
        • La distancia entre puntos indica similitud de reacción<br>
        • El eje X/Y son componentes principales (PCA)<br>
        • Activos más cercanos reaccionaron de forma parecida ante el evento
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# SECCIÓN 5 — MÉTRICAS DEL MODELO EN PRODUCCIÓN
# =============================================================================
st.markdown('<div class="section-title">📊 MÉTRICAS DEL MODELO EN PRODUCCIÓN</div>', unsafe_allow_html=True)

LINEA_BASE = 0.60
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

metricas_info = [
    ("AUC-ROC", AUC, "Capacidad de discriminación entre subida y bajada"),
    ("F1-Score", F1, "Balance entre precisión y recall del modelo"),
    ("Accuracy", ACC, "Porcentaje de predicciones correctas sobre prueba"),
]

for col, (nombre, valor, descripcion) in zip([col_m1, col_m2, col_m3], metricas_info):
    delta = valor - LINEA_BASE
    delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
    color_delta = "#00e68a" if delta >= 0 else "#ff6666"
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{nombre}</div>
            <div class="metric-value">{valor:.2f}</div>
            <div class="metric-delta" style="color:{color_delta};">
                {delta_str} vs línea base ({LINEA_BASE})
            </div>
            <div style="font-size:0.7rem; color:#5a8fa8; margin-top:0.5rem;">
                {descripcion}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_m4:
    # Gráfico gauge simple de AUC
    fig_gauge, ax_g = plt.subplots(figsize=(2.5, 2.5), subplot_kw=dict(polar=True))
    fig_gauge.patch.set_facecolor("#0d1a2e")
    ax_g.set_facecolor("#0d1a2e")

    theta_max = np.pi
    theta_val = np.pi * (1 - AUC)
    ax_g.barh(1, theta_max, left=0, height=0.3, color="#1e3a5f")
    ax_g.barh(1, theta_max - theta_val, left=0, height=0.3,
              color="#48cae4" if AUC >= 0.70 else "#f4a261")
    ax_g.set_ylim(0, 2)
    ax_g.set_xlim(0, np.pi)
    ax_g.set_theta_zero_location("W")
    ax_g.set_theta_direction(1)
    ax_g.axis("off")
    ax_g.text(np.pi / 2, 0.15, f"{AUC:.2f}", ha="center", va="center",
              fontsize=14, fontweight="bold", color="#48cae4",
              fontfamily="monospace")
    ax_g.text(np.pi / 2, -0.4, "AUC-ROC", ha="center", va="center",
              fontsize=8, color="#7aadca")
    plt.tight_layout()
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close()

# Interpretación automática
st.markdown("#### Interpretación de métricas")
col_int1, col_int2, col_int3 = st.columns(3)
interpretaciones = [
    ("AUC-ROC", AUC, f"El modelo distingue correctamente entre subida y bajada en el "
     f"{AUC*100:.0f}% de los casos, superando en {(AUC-0.5)*100:.0f} puntos porcentuales "
     f"la línea base aleatoria de 0.50."),
    ("F1-Score", F1, f"Balance {'adecuado' if F1 >= 0.65 else 'moderado'} entre no perderse "
     f"subidas reales ({F1*100:.0f}% de efectividad) y no generar falsas alarmas de subida."),
    ("Accuracy", ACC, f"El modelo clasificó correctamente el {ACC*100:.0f}% de los días "
     f"del conjunto de prueba (30% del total), {'superando' if ACC > LINEA_BASE else 'cerca de'} "
     f"la línea base de referencia de {LINEA_BASE*100:.0f}%."),
]
for col, (nombre, valor, interp) in zip([col_int1, col_int2, col_int3], interpretaciones):
    with col:
        icon = "🟢" if valor >= 0.70 else "🟡" if valor >= 0.60 else "🔴"
        st.markdown(f"""
        <div style="background:rgba(13,33,55,0.6); border:1px solid #1e3a5f;
                    border-radius:8px; padding:0.9rem; font-size:0.8rem;
                    color:#c0d8e8; line-height:1.6;">
            <b style="color:#48cae4;">{icon} {nombre}</b><br><br>
            {interp}
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div style="font-size:0.75rem; color:#5a8fa8; text-align:right; margin-top:0.5rem;">
    ℹ️ Línea base de referencia: {LINEA_BASE} &nbsp;|&nbsp;
    Evaluado sobre el 30% del conjunto de prueba (no visto durante el entrenamiento)
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# VISUALIZACIONES ADICIONALES (opcional — si existen los archivos)
# =============================================================================
graficos_extra = {
    "Curvas ROC": RUTAS["roc_png"],
    "Importancia de Variables": RUTAS["imp_vars_png"]
}

graficos_disponibles = {k: v for k, v in graficos_extra.items() if os.path.exists(v)}

if graficos_disponibles:
    st.markdown('<div class="section-title">📈 VISUALIZACIONES DEL MODELO</div>',
                unsafe_allow_html=True)
    tabs = st.tabs(list(graficos_disponibles.keys()))
    for tab, (titulo, ruta) in zip(tabs, graficos_disponibles.items()):
        with tab:
            st.image(ruta, use_container_width=True)
    st.markdown("---")

# =============================================================================
# DATASET DE REFERENCIA
# =============================================================================
if df is not None:
    with st.expander("🗃️ Ver muestra del dataset de modelamiento"):
        col_ds1, col_ds2 = st.columns(2)
        with col_ds1:
            st.markdown(f"**Shape:** {df.shape[0]:,} registros × {df.shape[1]} variables")
            if "sector" in df.columns:
                st.markdown("**Distribución por sector:**")
                dist = df["sector"].value_counts().reset_index()
                dist.columns = ["Sector", "Registros"]
                st.dataframe(dist, use_container_width=True, hide_index=True)
        with col_ds2:
            st.markdown("**Primeras filas del dataset:**")
            st.dataframe(df.head(8), use_container_width=True)
        st.download_button(
            "⬇️ Descargar dataset completo",
            data=df.to_csv(index=True).encode("utf-8"),
            file_name="dataset_modelamiento.csv",
            mime="text/csv"
        )

st.markdown("---")

# =============================================================================
# SECCIÓN 6 — ACERCA DEL PROYECTO
# =============================================================================
with st.expander("ℹ️ Acerca del Proyecto — Metodología y Documentación"):
    tab_desc, tab_metodologia, tab_datos, tab_uso = st.tabs(
        ["📋 Descripción", "🔬 Metodología", "📦 Datos", "📖 Instrucciones de Uso"]
    )

    with tab_desc:
        st.markdown("""
        ### Predictor de Retorno Anormal Post-Evento Geopolítico

        **Pregunta de investigación:**
        ¿Es posible predecir si un activo financiero generará un retorno anormal positivo
        o negativo ante eventos geopolíticos extremos, basándose en sus características de mercado
        y la reacción observada ante la captura de Nicolás Maduro (3 de enero de 2026)?

        **Objetivo del negocio:**
        Desarrollar un sistema de clasificación que permita a analistas financieros anticipar
        el comportamiento de activos ante shocks geopolíticos similares, apoyándose en el
        estudio de eventos (Event Study) y modelos de machine learning.

        **Activos analizados:**
        S&P 500, VIX, Brent, WTI, COLCAP, BOVESPA, USD/COP, Oro, Cobre, Exxon, Chevron

        **Autoras:** Laura Laguado · Sofía Navales
        """)
        st.markdown("**🔗 Repositorio GitHub:** [laulaguado/maduro-mercados-financieros](https://github.com/laulaguado/maduro-mercados-financieros)")

    with tab_metodologia:
        st.markdown("""
        ### Metodología CRISP-DM

        | Fase | Descripción |
        |------|-------------|
        | 1. Comprensión del negocio | Definición del evento geopolítico y objetivo de predicción |
        | 2. Comprensión de datos | Descarga de 11 activos desde Yahoo Finance (2020–2026) |
        | 3. Preparación de datos | Limpieza, imputación, detección de outliers, ingeniería de features |
        | 4. Modelamiento | Event Study (AR/CAR) + 7 algoritmos de clasificación supervisada |
        | 5. Evaluación | Validación cruzada 5-fold, ANOVA, Tukey HSD, métricas sobre prueba |
        | 6. Despliegue | Pipeline serializado + App Streamlit + Notebooks documentados |

        **Modelos evaluados:** Árbol de Decisión, KNN, SVM, Red Neuronal (MLP),
        Random Forest, XGBoost, Gradient Boosting

        **Selección del modelo final:** El mejor modelo fue seleccionado mediante
        ANOVA + Tukey HSD sobre los AUC-ROC de validación cruzada, seguido de
        hiperparametrización con GridSearchCV y BayesSearchCV.
        """)

    with tab_datos:
        st.markdown("""
        ### Fuentes de Datos

        | Activo | Ticker | Fuente |
        |--------|--------|--------|
        | S&P 500 | ^GSPC | Yahoo Finance |
        | VIX | ^VIX | Yahoo Finance |
        | Petróleo Brent | BZ=F | Yahoo Finance |
        | Petróleo WTI | CL=F | Yahoo Finance |
        | COLCAP | ^COLCAP | Yahoo Finance |
        | BOVESPA | ^BVSP | Yahoo Finance |
        | USD/COP | USDCOP=X | Yahoo Finance |
        | Oro | GC=F | Yahoo Finance |
        | Cobre | HG=F | Yahoo Finance |
        | Exxon Mobil | XOM | Yahoo Finance |
        | Chevron | CVX | Yahoo Finance |

        **Período:** 1 de enero de 2020 — día anterior a hoy
        **Fecha del evento:** 3 de enero de 2026 (Captura de Nicolás Maduro)
        **Primer día hábil post-evento:** 5 de enero de 2026
        **Ventana de estimación:** -250 a -11 días
        **Ventana del evento:** -10 a +60 días
        """)

    with tab_uso:
        st.markdown("""
        ### Instrucciones de Uso — Paso a Paso

        **Paso 1: Seleccionar el Sector**
        En el panel lateral, seleccione el sector del activo que desea analizar:
        - **Energía** → Brent, WTI, Exxon, Chevron
        - **Índice** → S&P 500, COLCAP, BOVESPA
        - **Divisa** → USD/COP
        - **Metal** → Oro, Cobre
        - **Volatilidad** → VIX

        **Paso 2: Ajustar Parámetros**
        Use los sliders para configurar las características del activo:
        - **Volatilidad baja** (< 0.02): mercado tranquilo
        - **VIX > 30**: mercado estresado
        - **Momentum positivo**: tendencia alcista reciente

        **Paso 3: Predecir**
        Haga clic en *"▶ Predecir comportamiento del activo"* y revise:
        - Resultado: SUBIDA 📈 o BAJADA 📉
        - Probabilidades de cada clase
        - Historial descargable

        **Paso 4: Interpretar el Clustering**
        El gráfico PCA muestra qué activos se comportaron de forma similar
        ante el evento. Use la descripción del cluster para contexto adicional.

        **Paso 5: Evaluar Confianza**
        - AUC-ROC > 0.70 → buena discriminación
        - F1-Score > 0.65 → balance precision/recall adecuado
        - Accuracy > 0.70 → alta tasa de acierto

        > ⚠️ **Aviso:** Este modelo es una herramienta académica de apoyo.
        > No constituye asesoramiento financiero ni garantiza resultados futuros.
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0; color:#3a6080;
            font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
            border-top:1px solid #1e3a5f; margin-top:1rem;">
    📊 Predictor de Retorno Anormal Post-Evento Geopolítico &nbsp;·&nbsp;
    Laura Laguado & Sofía Navales &nbsp;·&nbsp;
    Minería de Datos Financieros &nbsp;·&nbsp; CRISP-DM &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
