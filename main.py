from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import PricePredictor
from fastapi.responses import FileResponse

app = FastAPI(
    title="Mobile Price Classification API",
    description="API для прогнозування цінової категорії смартфона"
)

predictor = PricePredictor()

class PhoneFeatures(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/predict")
def predict_price(features: PhoneFeatures):
    features_dict = features.model_dump()
    
    predicted_class = predictor.predict_price(features_dict)
    
    class_names = {
        0: "0 (Бюджетний)",
        1: "1 (Середній)",
        2: "2 (Преміум)",
        3: "3 (Флагман)"
    }
    
    return {
        "predicted_class": predicted_class,
        "price_label": class_names.get(predicted_class, "Невідомо")
    }