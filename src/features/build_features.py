"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""

import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Maneja los valores faltantes.
       Puedes llenarlos con la mediana de la columna.
    2. Retorna el DataFrame limpio.
    """
    #****** 1. Duplicados ********
    #__________Explicitos________
    clean_data = df.drop_duplicates(keep='last')
    borrados = len(df) - len(clean_data)
    print(f"Se eliminaron {borrados} registros duplicados.")
    print(f"Registros restantes: {len(clean_data)}")
    #__________Logicos____________
    clean_data['income_rounded'] = clean_data['median_income'].round(2)
    columnas_logicas = ['longitude', 'latitude', 'housing_median_age', 'income_rounded']
    total_antes = len(clean_data)
    clean_data.drop_duplicates(subset=columnas_logicas, keep='last', inplace=True)
    clean_data.drop(columns=['income_rounded'], inplace=True)
    eliminados = total_antes - len(clean_data)
    print(f"--- Reporte de Duplicados Lógicos ---")
    print(f"Se eliminaron registros con valores iguales de ubicacion, antiguedad e ingresos: {eliminados}")
    print(f"Registros restantes: {len(clean_data)}")


    #****** 2. Valores ausentes **********
    total_filas = len(clean_data)
    filas_con_nulos = clean_data.isnull().any(axis=1).sum()
    porcentaje_nulos = (filas_con_nulos / total_filas) * 100

    print(f"Porcentaje de filas con nulos: {porcentaje_nulos:.2f}%")

    #______ Eliminar nulos si el porcentaje es menor al 3%________
    if porcentaje_nulos < 3:
        clean_data.dropna(inplace=True)
        print(f"Registros eliminados: {filas_con_nulos}")
    #______ Imputar nulos con mediana si es un porcentaje mayor_____
    else:
        columnas_numericas = clean_data.select_dtypes(include=['number']).columns
        for col in columnas_numericas:
            mediana = clean_data[col].median()
            clean_data[col].fillna(mediana, inplace=True)
        
        print(f"Se imputaron nulos con la mediana.")

    print(f"Registros finales en clean_data: {len(clean_data)}")
    print(f"Nulos restantes: {clean_data.isnull().sum().sum()}")

    #******* 3. Outliers **********
    # __________LIMITES, eliminar registros con techos artificiales_____________
    filas_antes = len(clean_data)
    clean_data = clean_data[clean_data['median_house_value'] < 500001]
    filas_eliminadas = filas_antes - len(clean_data)
    porcentaje_eliminado = (filas_eliminadas / filas_antes) * 100
    print(f"Filas eliminadas: {filas_eliminadas}")
    print(f"Porcentaje eliminado: {porcentaje_eliminado:.2f}%")

    # Outliers, rango intercuartil
    columnas_iqr = [
        'total_rooms',
        'total_bedrooms',
        'population',
        'households'
    ]

    filas_antes = len(clean_data)

    for col in columnas_iqr:
        Q1 = clean_data[col].quantile(0.25)
        Q3 = clean_data[col].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        clean_data = clean_data[
            (clean_data[col] >= limite_inferior) &
            (clean_data[col] <= limite_superior)
        ]

    clean_data.reset_index(drop=True, inplace=True)
    filas_eliminadas = filas_antes - len(clean_data)
    print("Filas originales:", filas_antes)
    print("Filas eliminadas:", filas_eliminadas)
    print("Filas restantes:", len(clean_data))

    #TRANSFORMACIÓN LOG para disminuir la cola 
    clean_data['median_income_log'] = np.log1p(clean_data['median_income'])

    #***** Variables Categoricas **************
    # Oceano
    ocean_mapping = {'INLAND': 0,'<1H OCEAN': 1,'NEAR OCEAN': 2,'NEAR BAY': 3,'ISLAND': 4}
    clean_data['ocean_ordinal'] = clean_data['ocean_proximity'].map(ocean_mapping)

    # Antiguedad
    bins_age = [0, 18, 35, np.inf]
    labels_age = ['Nueva', 'Media', 'Vieja']
    age_mapping = {'Nueva': 0, 'Media': 1, 'Vieja': 2}
    clean_data['age_ordinal'] = pd.cut(clean_data['housing_median_age'], 
                                    bins=bins_age, 
                                    labels=labels_age).map(age_mapping).astype(int)

    cols_to_drop = ['ocean_proximity', 'housing_median_age']
    clean_data.drop(columns=cols_to_drop, inplace=True)

    return clean_data

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Agrega nuevas variables derivando las existentes, por ejemplo:
       - 'rooms_per_household' = total_rooms / households
       - 'population_per_household' = population / households
       - 'bedrooms_per_room' = total_bedrooms / total_rooms
    2. Retorna el DataFrame enriquecido.
    """
    df["rooms_per_household"] = (df["total_rooms"] / df["households"]).round(2)
    df["bedrooms_per_room"] = (df["total_bedrooms"] / df["total_rooms"]).round(4)
    df['population_per_household'] = (df['population'] / df['households']).round(2)
    df["income_per_person"] = (df["median_income"] / df["population_per_household"]).round(4)

    df.drop(columns=['median_income'], inplace=True)

    return df

def preprocess_test(df_test: pd.DataFrame):
# --- Codificaciones Categóricas ---
    ocean_mapping = {'INLAND': 0, '<1H OCEAN': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4}
    df_test['ocean_ordinal'] = df_test['ocean_proximity'].map(ocean_mapping).astype(int)

    # --- Codificación de antiguedad ---
    bins_age = [0, 18, 35, np.inf]
    labels_age = ['Nueva', 'Media', 'Vieja']
    age_mapping = {'Nueva': 0, 'Media': 1, 'Vieja': 2}
    df_test['age_cat'] = pd.cut(df_test['housing_median_age'], bins=bins_age, labels=labels_age)
    df_test['age_ordinal'] = df_test['age_cat'].map(age_mapping).astype(int)

    # --- Ingeniería de Variables ---
    df_test['rooms_per_household'] = df_test['total_rooms'] / df_test['households']
    df_test['bedrooms_per_room'] = df_test['total_bedrooms'] / df_test['total_rooms']
    df_test['population_per_household'] = df_test['population'] / df_test['households']
    df_test['income_per_person'] = df_test['median_income'] / df_test['population']

    # --- Logaritmo y limpieza de columnas ---
    df_test['median_income_log'] = np.log1p(df_test['median_income'])

    #Guardar
    df_test.to_csv('../data/processed/test_processed.csv', index=False)
    

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que toma el DataFrame crudo y aplica limpieza y enriquecimiento.
    """
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)
    
    # IMPORTANTE: Aquí los alumnos deberían añadir codificación de variables categóricas
    # (ej. get_dummies para 'ocean_proximity') si no usan Pipelines de Scikit-Learn.
    
    return df_featured

if __name__ == "__main__":
    df = pd.read_csv(r'../data/interim/train_set.csv')
    df_procesado = preprocess_pipeline(df)
    df_procesado.to_csv('../data/processed/train_processed.csv', index=False)
    df_test = pd.read_csv(r'../data/interim/test_set.csv')
    preprocess_test(df_test)
    print("Módulo de feature engineering... (Falta el código!)")
