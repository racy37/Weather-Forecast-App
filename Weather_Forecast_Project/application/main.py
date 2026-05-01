from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # 新增：用于返回 HTML 文件
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Weather Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取当前文件的绝对路径，确保不管在哪里运行都能找到模型和网页
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'weather_rf_model.pkl')
HTML_PATH = os.path.join(BASE_DIR, 'static', 'index.html')

try:
    model = joblib.load(MODEL_PATH)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败，请检查文件路径: {e}")

class WeatherInput(BaseModel):
    temperature_2m_max: float
    temperature_2m_min: float
    precipitation_sum: float
    wind_speed_10m_max: float

# 新增路由：当用户访问主页 "/" 时，直接返回你的漂亮的 HTML 网页
@app.get("/")
def serve_frontend():
    return FileResponse(HTML_PATH)

@app.post("/predict")
def predict_tomorrow_weather(data: WeatherInput):
    input_data = {
        'temperature_2m_max': [data.temperature_2m_max],
        'temperature_2m_min': [data.temperature_2m_min],
        'precipitation_sum': [data.precipitation_sum],
        'wind_speed_10m_max': [data.wind_speed_10m_max]
    }
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]
    return {
        "status": "success",
        "predicted_tomorrow_max_temp": round(prediction, 1)
    }