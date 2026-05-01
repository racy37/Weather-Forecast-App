import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

print("正在从 Open-Meteo 获取香港历史天气数据...")

# 设定请求 API 的参数 (纬度 22.28, 经度 114.16 是香港)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 22.28,
    "longitude": 114.16,
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
    "timezone": "Asia/Hong_Kong"
}

response = requests.get(url, params=params)
data = response.json()

# 将 JSON 数据转换为 Pandas DataFrame (数据表)
df = pd.DataFrame(data['daily'])
df['time'] = pd.to_datetime(df['time']) # 将时间列转为标准时间格式
df.set_index('time', inplace=True)      # 把时间设置为索引

print("数据获取成功！前5行数据如下：")
print(df.head())

# 1. 构造目标变量 (Target)：明天的最高气温
# 我们把当天的最高温向上移动一格 (shift(-1))，作为今天的预测目标
df['target_tomorrow_max_temp'] = df['temperature_2m_max'].shift(-1)

# 2. 数据清洗：处理缺失值
# 因为最后一天没有“明天”的数据，它会变成 NaN (空值)，我们需要把它删掉
df.dropna(inplace=True)

# 3. 查看处理后的数据形态
print("清洗后的数据维度 (行数, 列数):", df.shape)
print("\n准备就绪的训练数据样本：")
print(df[['temperature_2m_max', 'precipitation_sum', 'target_tomorrow_max_temp']].head())




print("\n\n====== 开始第二步：模型训练与保存 ======")

# 1. 定义特征 (X) 和 目标变量 (y)
# X 是今天的各项天气指标，y 是我们想要预测说明天的最高气温
features = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']
X = df[features]
y = df['target_tomorrow_max_temp']

# 2. 划分训练集和测试集
# 我们拿出 80% 的日子用来给 AI 学习，剩下的 20% 藏起来用来考试（测试模型准不准）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape[0]} 天, 测试集大小: {X_test.shape[0]} 天")

# 3. 训练随机森林模型
print("正在训练随机森林模型 (Random Forest)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 模型评估
# 让训练好的模型做那 20% 隐藏数据的卷子
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"模型评估完成！均方误差 (MSE): {mse:.2f}") 
# MSE 越小越好，说明预测温度和真实温度越接近

# 打印 5 条数据，直观感受一下 AI 的预测能力
print("\n--- 模型预测效果抽查 ---")
results = pd.DataFrame({
    '实际明天最高温': y_test.values[:5], 
    '模型预测最高温': predictions[:5]
})
print(results.round(1)) # 保留一位小数

# 5. 将“AI的大脑”保存为文件，这是后续网站能做预测的核心！
model_filename = 'weather_rf_model.pkl'
joblib.dump(model, model_filename)
print(f"\n太棒了！模型已成功保存为文件: {model_filename}")