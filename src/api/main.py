"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Inicializamos la app
app = FastAPI(title="API de Predicción de Precios de Vivienda (California)", version="1.0")

# INSTRUCCIONES: Define el esquema de datos esperado por la API (Las variables X que usa tu modelo)
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # Necesario para el mapeo ordinal

# Variable global para cargar el modelo
# IMPORTANTE: Asegúrate de guardar tu modelo en "models/best_model.pkl" o ajusta la ruta
model = None
scaler = None

@app.on_event("startup")
def load_assets():
    """Carga el modelo y el escalador al iniciar el servidor."""
    global model, scaler
    try:
        model = joblib.load("models/housing_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        print("Modelo y escalador cargados exitosamente.")
    except Exception as e:
        print(f"Error al cargar archivos: {e}")

@app.get("/")
def home():
    return {"mensaje": "API activa. Usa el endpoint /predict para obtener estimaciones."}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    
    # Convertir entrada a DataFrame para manipularlo fácilmente
    data = pd.DataFrame([features.dict()])

    # --- Transformaciones de Categorías ---
    ocean_mapping = {'INLAND': 0, '<1H OCEAN': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4}
    data['ocean_ordinal'] = data['ocean_proximity'].map(ocean_mapping).fillna(0).astype(int)

    bins_age = [0, 18, 35, np.inf]
    labels_age = [0, 1, 2] # Usamos directamente el mapeo (Nueva=0, Media=1, Vieja=2)
    data['age_ordinal'] = pd.cut(data['housing_median_age'], bins=bins_age, labels=labels_age).astype(int)

    # --- Ingeniería de Variables  ---
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    data['income_per_person'] = data['median_income'] / data['population']

    # ---  Logaritmo de Ingresos ---
    data['median_income_log'] = np.log1p(data['median_income'])

    # ORDEN exacto de las 13 columnas de entrada
    cols_input = [
        'longitude', 'latitude', 'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income_log', 'ocean_ordinal', 'age_ordinal', 
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'income_per_person'
    ]
    
    X_final = data[cols_input]

    # Escalar y Predecir
    try:
        X_scaled = scaler.transform(X_final)
        prediction = model.predict(X_scaled)
        
        return {
            "status": "success",
            "predicted_price": float(np.round(prediction[0], 2)),
            "currency": "USD"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")

# Instrucciones para correr la API localmente:
# En la terminal, ejecuta:
# uvicorn src.api.main:app --reload



# Variables globales que se llenarán al iniciar
# model = None
# scaler = None
# expected_features = None

# @app.on_event("startup")
# def load_assets():
#     """Carga el diccionario unificado y desempaqueta sus componentes."""
#     global model, scaler, expected_features
#     try:
#         # Cargamos el archivo único
#         checkpoint = joblib.load("models/full_pipeline.joblib")
        
#         # Desempaquetamos según tu estructura
#         model = checkpoint["model"]
#         scaler = checkpoint["scaler"]
#         expected_features = checkpoint["features"]
        
#         print("Activos desempaquetados correctamente: Modelo, Scaler y Features listos.")
#     except Exception as e:
#         print(f"Error crítico al cargar el archivo unificado: {e}")

# @app.post("/predict")
# def predict_price(features: HousingFeatures):
#     if model is None or scaler is None:
#         raise HTTPException(status_code=503, detail="Modelo no cargado en el servidor.")
    
#     # 2. Convertir a DataFrame
#     data = pd.DataFrame([features.dict()])

#     # --- Transformaciones de Categorías ---
#     ocean_mapping = {'INLAND': 0, '<1H OCEAN': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4}
#     data['ocean_ordinal'] = data['ocean_proximity'].map(ocean_mapping).fillna(0).astype(int)

#     bins_age = [0, 18, 35, np.inf]
#     data['age_ordinal'] = pd.cut(data['housing_median_age'], 
#                                  bins=bins_age, 
#                                  labels=[0, 1, 2]).astype(int)

#     # --- Ingeniería de Variables ---
#     data['rooms_per_household'] = data['total_rooms'] / data['households']
#     data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
#     data['population_per_household'] = data['population'] / data['households']
#     data['income_per_person'] = data['median_income'] / data['population']

#     # --- Logaritmo de Ingresos ---
#     data['median_income_log'] = np.log1p(data['median_income'])

#     # 3. Uso de la variable 'expected_features' cargada del joblib
#     # Esto asegura que el orden de las columnas sea SIEMPRE el correcto
#     try:
#         X_final = data[expected_features]
#         X_scaled = scaler.transform(X_final)
#         prediction = model.predict(X_scaled)
        
#         return {
#             "predicted_price": float(np.round(prediction[0], 2)),
#             "model_version": "RandomForest_Optimized",
#             "features_used": expected_features
#         }
#     except KeyError as e:
#         raise HTTPException(status_code=400, 
#                             detail=f"Falta una columna requerida por el modelo: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")