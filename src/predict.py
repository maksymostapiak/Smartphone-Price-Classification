import joblib
import pandas as pd
import os

class PricePredictor:
    def __init__(self):
        """Ініціалізується один раз при старті API."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'model.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_price(self, phone_features: dict) -> int:
        """
        Приймає словник з характеристиками, масштабує і повертає клас ціни.
        """
        df_input = pd.DataFrame([phone_features])
        
        scaled_features = self.scaler.transform(df_input)
        
        prediction = self.model.predict(scaled_features)
        return int(prediction[0])