import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Завантажує та очищає дані."""
    df = pd.read_csv(filepath)
    
    missing_values = df.isnull().sum()
    
    if missing_values.sum() == 0:
        print("У датасеті немає пропущених значень.")
    else:
        df = df.fillna(df.median())
        print(f"Залишилось пропусків: {df.isnull().sum().sum()}")
        
    return df