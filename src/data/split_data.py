"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    #Carga de datos
    raw_data = pd.read_csv(raw_data_path)
    
    # Estratificación por caterogías (median_income) para asegurar que el split sea representativo.
    # Dividimos por 1.5 para limitar el número de categorías
    raw_data["income_cat"] = np.ceil(raw_data["median_income"] / 1.5)
    # Agrupamos todas las categorías mayores a 5 en la categoría 5.0
    raw_data["income_cat"].where(raw_data["income_cat"] < 5, 5.0, inplace=True)

    # Partición Estratificada
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(raw_data, raw_data["income_cat"]):
        strat_train_set = raw_data.loc[train_index]
        strat_test_set = raw_data.loc[test_index]

    # Quitar la columna temporal 'income_cat' para tener los datos originales
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Guardar datos en la ruta interim
    os.makedirs(interim_data_path, exist_ok=True)
    
    train_path = os.path.join(interim_data_path, "train_set.csv")
    test_path = os.path.join(interim_data_path, "test_set.csv")
    
    strat_train_set.to_csv(train_path, index=False)
    strat_test_set.to_csv(test_path, index=False)

if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
    print("Script para dividir datos... (Falta el código!)")
