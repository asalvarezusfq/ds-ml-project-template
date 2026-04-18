"""
Script de entrenamiento para el modelo final y evaluación básica.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math

# IMPORTANTE: Se debe importar los algoritmos que quieran usar, por ejemplo:
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

def train_best_model(processed_train_data_path: str, model_save_path: str):
    """
    INSTRUCCIONES:
    1. Carga los datos de entrenamiento procesados (que ya pasaron por `build_features.py`).
    2. Separa las características (X) de la etiqueta a predecir (y = 'median_house_value').
    3. Instancia tu mejor modelo encontrado después de la fase de experimentación y "fine tuning"
       (Por ejemplo: RandomForestRegressor con los mejores hiperparámetros).
    4. Entrena el modelo haciendo fit(X, y).
    5. Guarda el modelo entrenado en `model_save_path` (ej. 'models/best_model.pkl') usando joblib.dump().
    """
    path_root = "../" + processed_train_data_path
    df = pd.read_csv(path_root)
    # 2. Preparación de X (características) y y (objetivo)
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    # Escalamiento de datos
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # División en entrenamiento y validación (80-20)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Mejor Modelo 
    model = RandomForestRegressor(
    bootstrap=False,
    max_depth=None,
    max_features=8,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
    model.fit(X_train, y_train)
    #Guardar modelo
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": X_train.columns.tolist()
    }, model_save_path)



def evaluate_model(model_path: str, processed_test_data_path: str):
    """
    INSTRUCCIONES:
    1. Carga el modelo guardado con joblib.load().
    2. Carga los datos de prueba preprocesados.
    3. Genera predicciones (y_pred) sobre los datos de prueba usando predict().
    4. Compara y_pred con las etiquetas reales calculando el RMSE y repórtalo en la terminal.
    """
    # Carga de modelo
    best_model = joblib.load(model_path)
    model = best_model["model"]
    scaler = best_model["scaler"]
    features = best_model["features"]

    # Carga datos de prueba preprocesados.
    path_root = "../" + processed_test_data_path
    df_test = pd.read_csv(path_root)
    X_test = df_test[features]
    y_test = df_test['median_house_value']
    # Escalar
    X_test_scaled = scaler.transform(X_test)
    # Predicion
    y_pred = model.predict(X_test_scaled)
    # RMSE
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")


    pass

if __name__ == "__main__":
    PROCESSED_TRAIN_PATH = "data/processed/train_processed.csv"
    PROCESSED_TEST_PATH = "data/processed/test_processed.csv"
    MODEL_OUTPUT_PATH = "models/best_model.pkl"
    train_best_model(PROCESSED_TRAIN_PATH, MODEL_OUTPUT_PATH)
    evaluate_model(MODEL_OUTPUT_PATH, PROCESSED_TEST_PATH)
    print("Script de entrenamiento final... (Falta el código!)")
