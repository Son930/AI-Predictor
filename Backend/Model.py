from prophet import Prophet
import pandas as pd
import main

# Load your stock data (Date, Close Price)
df = pd.read_csv()
df.rename(columns={"date": "ds", "close": "y"}, inplace=True)

# Train Prophet
model = Prophet()
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
