#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para el entrenamiento de modelos predictivos.
Parte del proyecto de análisis del impacto de la captura de Nicolás Maduro
en los mercados financieros globales.

Autoras: Laura Laguado y Sofía Navales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
import statsmodels.stats.multicomp as mc
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import os
import logging
from tabulate import tabulate

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)

# Configuración global de los modelos
MODELOS = {
    'arbol_decision': DecisionTreeClassifier(random_state=42),
    'knn': KNeighborsClassifier(),
    'svm': SVC(probability=True, random_state=42),
    'red_neuronal': MLPClassifier(random_state=42, max_iter=500),
    'random_forest': RandomForestClassifier(random_state=42),
    'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'gradient_boosting': GradientBoostingClassifier(random_state=42)
}

# Grids de hiperparámetros para GridSearchCV
GRIDS_HIPERPARAMETROS = {
    'arbol_decision': {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 11, 15],
        'metric': ['euclidean', 'manhattan']
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'red_neuronal': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01]
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'max_features': ['sqrt', 'log2']
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
}

# Espacios de búsqueda para BayesSearchCV
ESPACIOS_BAYES = {
    'arbol_decision': {
        'max_depth': Integer(2, 15),
        'min_samples_split': Integer(2, 20),
        'criterion': Categorical(['gini', 'entropy'])
    },
    'knn': {
        'n_neighbors': Integer(1, 20),
        'metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
    },
    'svm': {
        'C': Real(0.01, 100, prior='log-uniform'),
        'kernel': Categorical(['rbf', 'linear']),
        'gamma': Real(0.001, 10, prior='log-uniform')
    },
    'red_neuronal': {
        'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)]),
        'activation': Categorical(['relu', 'tanh']),
        'learning_rate_init': Real(0.0001, 0.1, prior='log-uniform')
    },
    'random_forest': {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 15),
        'max_features': Categorical(['sqrt', 'log2', None])
    },
    'xgboost': {
        'n_estimators': Integer(50, 300),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(2, 10),
        'subsample': Real(0.5, 1.0)
    },
    'gradient_boosting': {
        'n_estimators': Integer(50, 300),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(2, 10),
        'subsample': Real(0.5, 1.0)
    }
}


def dividir_datos(X, y, test_size=0.30, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba con estratificación.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Variable objetivo.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla para reproducibilidad.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    
    Example:
        >>> X_train, X_test, y_train, y_test = dividir_datos(X, y)
        >>> print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    """
    print("\n" + "="*80)
    print(f"DIVISIÓN DE DATOS (TEST_SIZE={test_size})")
    print("="*80)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Imprimir shapes
    print(f"\nShape de X_train: {X_train.shape}")
    print(f"Shape de X_test: {X_test.shape}")
    
    # Imprimir distribución de clases
    print("\nDistribución de clases en entrenamiento:")
    print(pd.Series(y_train).value_counts(normalize=True).round(4) * 100)
    
    print("\nDistribución de clases en prueba:")
    print(pd.Series(y_test).value_counts(normalize=True).round(4) * 100)
    
    return X_train, X_test, y_train, y_test


def aplicar_smote_si_necesario(X_train, y_train, umbral_desbalance=0.40):
    """
    Aplica SMOTE si la proporción de la clase minoritaria es menor que el umbral.
    
    Args:
        X_train (pandas.DataFrame): Features de entrenamiento.
        y_train (pandas.Series): Variable objetivo de entrenamiento.
        umbral_desbalance (float): Umbral para considerar desbalance.
    
    Returns:
        tuple: (X_train_bal, y_train_bal)
    
    Example:
        >>> X_train_bal, y_train_bal = aplicar_smote_si_necesario(X_train, y_train)
        >>> print(pd.Series(y_train_bal).value_counts(normalize=True))
    """
    print("\n" + "="*80)
    print(f"EVALUACIÓN DE BALANCE DE CLASES (UMBRAL={umbral_desbalance})")
    print("="*80)
    
    # Calcular proporción de la clase minoritaria
    conteo = pd.Series(y_train).value_counts()
    clase_min = conteo.idxmin()
    proporcion_min = conteo[clase_min] / len(y_train)
    
    print(f"\nProporción de la clase minoritaria ({clase_min}): {proporcion_min:.4f}")
    
    # Verificar si se necesita balanceo
    if proporcion_min < umbral_desbalance:
        print(f"Proporción < {umbral_desbalance}, aplicando SMOTE...")
        
        # Aplicar SMOTE
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        
        # Imprimir distribución después de SMOTE
        print("\nDistribución de clases después de SMOTE:")
        print(pd.Series(y_train_bal).value_counts(normalize=True).round(4) * 100)
        
        return X_train_bal, y_train_bal
    else:
        print(f"Proporción >= {umbral_desbalance}, no se requiere balanceo")
        return X_train, y_train


def entrenar_con_validacion_cruzada(X_train, y_train, n_folds=10):
    """
    Entrena modelos con validación cruzada estratificada.
    
    Args:
        X_train (pandas.DataFrame): Features de entrenamiento.
        y_train (pandas.Series): Variable objetivo de entrenamiento.
        n_folds (int): Número de folds para validación cruzada.
    
    Returns:
        dict: Resultados por modelo y por fold.
    
    Example:
        >>> resultados = entrenar_con_validacion_cruzada(X_train, y_train)
        >>> print(resultados['arbol_decision']['auc_mean'])
    """
    print("\n" + "="*80)
    print(f"ENTRENAMIENTO CON VALIDACIÓN CRUZADA ({n_folds}-FOLD)")
    print("="*80)
    
    # Crear validación cruzada estratificada
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Diccionario para almacenar resultados
    resultados = {}
    
    # Para cada modelo
    for nombre_modelo, modelo in MODELOS.items():
        print(f"\nEntrenando {nombre_modelo}...")
        
        # Listas para almacenar métricas por fold
        aucs = []
        f1s = []
        accuracies = []
        precisions = []
        recalls = []
        
        # Para cada fold
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            # Dividir datos
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Entrenar modelo
            modelo.fit(X_fold_train, y_fold_train)
            
            # Predecir probabilidades
            y_proba = modelo.predict_proba(X_fold_val)[:, 1]
            
            # Predecir clases
            y_pred = modelo.predict(X_fold_val)
            
            # Calcular métricas
            auc = roc_auc_score(y_fold_val, y_proba)
            f1 = f1_score(y_fold_val, y_pred)
            accuracy = accuracy_score(y_fold_val, y_pred)
            precision = precision_score(y_fold_val, y_pred)
            recall = recall_score(y_fold_val, y_pred)
            
            # Almacenar métricas
            aucs.append(auc)
            f1s.append(f1)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            
            print(f"  Fold {i+1}: AUC={auc:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")
        
        # Calcular estadísticas
        resultados[nombre_modelo] = {
            'auc': aucs,
            'f1': f1s,
            'accuracy': accuracies,
            'precision': precisions,
            'recall': recalls,
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls)
        }
    
    # Imprimir tabla resumen
    print("\nRESUMEN DE RESULTADOS:")
    tabla = []
    for nombre_modelo, res in resultados.items():
        fila = [
            nombre_modelo,
            f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f}",
            f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}",
            f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}",
            f"{res['precision_mean']:.4f} ± {res['precision_std']:.4f}",
            f"{res['recall_mean']:.4f} ± {res['recall_std']:.4f}"
        ]
        tabla.append(fila)
    
    headers = ["Modelo", "AUC-ROC", "F1", "Accuracy", "Precision", "Recall"]
    print(tabulate(tabla, headers=headers, tablefmt="grid"))
    
    return resultados


def aplicar_anova_tukey(resultados_cv):
    """
    Aplica ANOVA y test de Tukey para comparar modelos.
    
    Args:
        resultados_cv (dict): Resultados de validación cruzada.
    
    Returns:
        list: Lista con los 3 mejores modelos por AUC-ROC promedio.
    
    Example:
        >>> mejores_modelos = aplicar_anova_tukey(resultados)
        >>> print(mejores_modelos)
    """
    print("\n" + "="*80)
    print("COMPARACIÓN ESTADÍSTICA DE MODELOS (ANOVA + TUKEY)")
    print("="*80)
    
    # Extraer AUC-ROC por fold de cada modelo
    aucs_por_modelo = {}
    for nombre_modelo, res in resultados_cv.items():
        aucs_por_modelo[nombre_modelo] = res['auc']
    
    # Preparar datos para ANOVA
    modelo_nombres = []
    auc_valores = []
    
    for nombre_modelo, aucs in aucs_por_modelo.items():
        for auc in aucs:
            modelo_nombres.append(nombre_modelo)
            auc_valores.append(auc)
    
    # Crear DataFrame
    df_anova = pd.DataFrame({
        'modelo': modelo_nombres,
        'auc': auc_valores
    })
    
    # Aplicar ANOVA
    modelos = df_anova['modelo'].unique()
    grupos = [df_anova.loc[df_anova['modelo'] == modelo, 'auc'].values for modelo in modelos]
    
    f_stat, p_valor = stats.f_oneway(*grupos)
    
    print(f"\nResultados ANOVA:")
    print(f"F-estadístico: {f_stat:.4f}")
    print(f"p-valor: {p_valor:.4f}")
    
    # Aplicar Tukey HSD si ANOVA es significativo
    if p_valor < 0.05:
        print("\np-valor < 0.05: Existen diferencias significativas entre modelos")
        print("\nAplicando test de Tukey HSD...")
        
        # Aplicar Tukey
        tukey = mc.MultiComparison(df_anova['auc'], df_anova['modelo'])
        resultado_tukey = tukey.tukeyhsd()
        
        # Imprimir resultados
        print("\nResultados Tukey HSD:")
        print(resultado_tukey)
        
        # Identificar pares significativamente diferentes
        pares_sig = []
        for i, p in enumerate(resultado_tukey.pvalues):
            if p < 0.05:
                par = (resultado_tukey.groupsunique[resultado_tukey.data[i, 0]],
                       resultado_tukey.groupsunique[resultado_tukey.data[i, 1]])
                pares_sig.append(par)
        
        if pares_sig:
            print("\nPares con diferencias estadísticamente significativas:")
            for par in pares_sig:
                print(f"- {par[0]} vs {par[1]}")
        else:
            print("\nNo se encontraron pares con diferencias estadísticamente significativas")
    else:
        print("\np-valor >= 0.05: No hay evidencia suficiente para afirmar diferencias entre modelos")
    
    # Seleccionar los 3 mejores modelos por AUC-ROC promedio
    auc_promedios = {nombre: res['auc_mean'] for nombre, res in resultados_cv.items()}
    mejores_3 = sorted(auc_promedios.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("\nLos 3 mejores modelos por AUC-ROC promedio:")
    for i, (nombre, auc) in enumerate(mejores_3):
        print(f"{i+1}. {nombre}: {auc:.4f}")
    
    return [nombre for nombre, _ in mejores_3]


def hiperparametrizar_modelos(X_train, y_train, mejores_3_modelos):
    """
    Optimiza hiperparámetros para los mejores modelos.
    
    Args:
        X_train (pandas.DataFrame): Features de entrenamiento.
        y_train (pandas.Series): Variable objetivo de entrenamiento.
        mejores_3_modelos (list): Lista con los nombres de los 3 mejores modelos.
    
    Returns:
        dict: Diccionario con los 3 modelos hiperparametrizados.
    
    Example:
        >>> modelos_optimos = hiperparametrizar_modelos(X_train, y_train, mejores_modelos)
        >>> print(modelos_optimos['xgboost'].get_params())
    """
    print("\n" + "="*80)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*80)
    
    modelos_optimos = {}
    
    for nombre_modelo in mejores_3_modelos:
        print(f"\nOptimizando {nombre_modelo}...")
        
        # Obtener modelo base
        modelo_base = MODELOS[nombre_modelo]
        
        # GridSearchCV
        print("\n1. Aplicando GridSearchCV...")
        grid = GridSearchCV(
            estimator=modelo_base,
            param_grid=GRIDS_HIPERPARAMETROS[nombre_modelo],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        mejor_score_grid = grid.best_score_
        mejores_params_grid = grid.best_params_
        
        print(f"  Mejor AUC-ROC: {mejor_score_grid:.4f}")
        print(f"  Mejores parámetros: {mejores_params_grid}")
        
        # BayesSearchCV
        print("\n2. Aplicando BayesSearchCV...")
        bayes = BayesSearchCV(
            estimator=modelo_base,
            search_spaces=ESPACIOS_BAYES[nombre_modelo],
            n_iter=20,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        bayes.fit(X_train, y_train)
        
        mejor_score_bayes = bayes.best_score_
        mejores_params_bayes = bayes.best_params_
        
        print(f"  Mejor AUC-ROC: {mejor_score_bayes:.4f}")
        print(f"  Mejores parámetros: {mejores_params_bayes}")
        
        # Seleccionar el mejor entre Grid y Bayes
        if mejor_score_grid >= mejor_score_bayes:
            print("\nSeleccionando modelo de GridSearchCV (mejor AUC-ROC)")
            modelo_optimo = grid.best_estimator_
            mejor_score = mejor_score_grid
            mejores_params = mejores_params_grid
        else:
            print("\nSeleccionando modelo de BayesSearchCV (mejor AUC-ROC)")
            modelo_optimo = bayes.best_estimator_
            mejor_score = mejor_score_bayes
            mejores_params = mejores_params_bayes
        
        print(f"AUC-ROC final: {mejor_score:.4f}")
        print(f"Parámetros finales: {mejores_params}")
        
        # Guardar modelo optimizado
        modelos_optimos[nombre_modelo] = modelo_optimo
    
    return modelos_optimos


def construir_pipeline_final(mejor_modelo, scaler):
    """
    Construye y entrena un pipeline completo con el mejor modelo.
    
    Args:
        mejor_modelo (estimator): Mejor modelo seleccionado.
        scaler (StandardScaler): Escalador entrenado.
    
    Returns:
        sklearn.pipeline.Pipeline: Pipeline entrenado.
    
    Example:
        >>> pipeline = construir_pipeline_final(mejor_modelo, scaler)
        >>> y_pred = pipeline.predict(X_test)
    """
    print("\n" + "="*80)
    print("CONSTRUCCIÓN DEL PIPELINE FINAL")
    print("="*80)
    
    # Construir pipeline
    pipeline = Pipeline([
        ('imputador', SimpleImputer(strategy='median')),
        ('escalador', scaler),
        ('modelo', mejor_modelo)
    ])
    
    # Crear directorio para modelos si no existe
    directorio = os.path.join("models")
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Guardar pipeline
    ruta_modelo = os.path.join(directorio, "modelo_final.pkl")
    joblib.dump(pipeline, ruta_modelo)
    
    print(f"\nPipeline guardado en: {ruta_modelo}")
    print("\nComponentes del pipeline:")
    print(f"1. Imputador: SimpleImputer(strategy='median')")
    print(f"2. Escalador: {type(scaler).__name__}")
    print(f"3. Modelo: {type(mejor_modelo).__name__}")
    
    return pipeline


if __name__ == "__main__":
    # Ejemplo de uso del módulo
    import os
    from preprocessing import estandarizar_zscore
    
    # Cargar datos de ejemplo
    try:
        ruta_datos = os.path.join("data", "processed", "dataset_modelamiento.csv")
        df = pd.read_csv(ruta_datos, index_col=0)
        
        # Separar features y target
        X = df.drop(['target_BRENT'], axis=1)  # Ejemplo con BRENT como target
        y = df['target_BRENT']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = dividir_datos(X, y)
        
        # Aplicar SMOTE si es necesario
        X_train_bal, y_train_bal = aplicar_smote_si_necesario(X_train, y_train)
        
        # Estandarizar datos
        X_train_std, scaler = estandarizar_zscore(X_train_bal)
        
        # Entrenar con validación cruzada
        resultados = entrenar_con_validacion_cruzada(X_train_std, y_train_bal)
        
        # Aplicar ANOVA y Tukey
        mejores_modelos = aplicar_anova_tukey(resultados)
        
        # Hiperparametrizar modelos
        modelos_optimos = hiperparametrizar_modelos(X_train_std, y_train_bal, mejores_modelos)
        
        # Seleccionar el mejor modelo
        mejor_nombre = mejores_modelos[0]  # El primero es el mejor
        mejor_modelo = modelos_optimos[mejor_nombre]
        
        # Construir pipeline final
        pipeline = construir_pipeline_final(mejor_modelo, scaler)
        
        print("\nProceso completado con éxito.")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_datos}")
        print("Ejecute primero los módulos anteriores para generar los datos necesarios.")