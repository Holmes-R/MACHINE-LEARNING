import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:,.2f}'.format

data = pd.read_csv('cyber_data.csv')
country_data = data[data['Country'] == 'Argentine Republic'].copy()

attack_types = ['Spam', 'Ransomware', 'Local Infection', 'Exploit', 
                'Malicious Mail', 'Network Attack', 'On Demand Scan', 'Web Threat']

forecast_steps = 60  
forecasts = {}

arima_order = (5, 1, 4)  

for attack_type in attack_types:
    print(f"\nRunningforecast for {attack_type}...")

    ts_data = country_data[attack_type].dropna()

    if len(ts_data) > 10:  
        try:
            train_size = int(len(ts_data) * 0.9)
            train, test = ts_data[:train_size], ts_data[train_size:]

            history = list(train)
            rolling_predictions = []

            for t in range(len(test)):
                model = ARIMA(history, order=arima_order)
                model_fit = model.fit()
                
                yhat = model_fit.forecast(steps=1)[0]
                rolling_predictions.append(yhat)
                
                history.append(test.iloc[t])

            mse = mean_squared_error(test, rolling_predictions)
            mae = mean_absolute_error(test, rolling_predictions)
            rmse = math.sqrt(mse)

            print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}')

            model = ARIMA(ts_data, order=arima_order)
            model_fit = model.fit()
            extended_forecast = model_fit.forecast(steps=forecast_steps)

            forecasts[attack_type] = extended_forecast

        except Exception as e:
            print(f"Error running  forecast for {attack_type}: {e}")
    else:
        print(f"Not enough data for {attack_type} (less than 10 points). Skipping.")

plt.figure(figsize=(12, 8))
time_range = pd.date_range(start='2025-01-01', periods=forecast_steps, freq='M')

for attack_type, forecast in forecasts.items():
    plt.plot(time_range, forecast, label=f"{attack_type} - Forecast", linestyle='--', linewidth=1.5)

plt.title("Cyber Attack Types - Rolling ARIMA Forecast (2025-2030)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True)
plt.show()
