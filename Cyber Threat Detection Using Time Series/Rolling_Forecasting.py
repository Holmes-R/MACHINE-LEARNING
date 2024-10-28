import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

"""
This script performs a rolling ARIMA forecast on cyber attack probabilities for a specified attack type
in the dataset pertaining to the 'United States of America'. It computes the rolling predictions using the
ARIMA model and plots the forecasted probabilities for the next five years.

Procedure:
1. Load the dataset from a CSV file named 'cyber_data.csv'.
2. Filter the data to focus on the 'United States of America'.
3. Specify the attack types to forecast (currently only 'Spam').
4. For each attack type:
    a. Check if there is sufficient data (more than 10 points).
    b. Split the data into training (90%) and testing (10%) sets.
    c. Fit an ARIMA model to the training data, make rolling forecasts, and store predictions.
    d. Calculate evaluation metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
5. Create a date range for the next 5 years (60 months).
6. Plot the rolling predictions against the date range, labeling the axes and adding a title.

Usage:
    Ensure the dataset 'cyber_data.csv' is present in the working directory before running the script.
    Modify the `attack_types` list to include other attack types as needed for forecasting.

Outputs:
    - Prints MSE, MAE, and RMSE for each attack type forecasted.
    - Displays a plot of the rolling forecasted probabilities for the specified attack type over the next 5 years.

Note:
    This script is configured for only one attack type ('Spam') but can be modified to include more
    attack types by adjusting the `attack_types` list.
"""

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:,.2f}'.format

data = pd.read_csv('cyber_data.csv')

country_data = data[data['Country'] == 'United States of America'].copy()

attack_types = ['Spam']

forecasts = {}


for attack_type in attack_types:
    print(f"\nRunning rolling forecast for {attack_type}...")

    ts_data = country_data[attack_type].dropna()  

    if len(ts_data) > 10:  
        try:
            train_size = int(len(ts_data) * 0.9)
            train, test = ts_data[:train_size], ts_data[train_size:]

            history = list(train)
            rolling_predictions = []

            for t in range(len(test)):
                model = ARIMA(history, order=(1, 1, 0)) 
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
                rolling_predictions.append(yhat)
                history.append(test.iloc[t]) 

            forecasts[attack_type] = rolling_predictions

            mse = mean_squared_error(test, rolling_predictions)
            mae = mean_absolute_error(test, rolling_predictions)
            rmse = math.sqrt(mse)

            print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}')

        except Exception as e:
            print(f"Error running rolling forecast for {attack_type}: {e}")
    else:
        print(f"Not enough data for {attack_type} (less than 10 points). Skipping.")

plt.figure(figsize=(12, 8))

date_range = pd.date_range(start=pd.Timestamp.today(), periods=60, freq='M')

for attack_type, rolling_predictions in forecasts.items():
    rolling_index = range(len(date_range) - len(rolling_predictions), len(date_range))
    plt.plot(date_range[rolling_index], rolling_predictions, label=f"{attack_type} - Rolling Predictions", linestyle='--', linewidth=1.5)

plt.title("Cyber Attack Types - Rolling ARIMA Forecast (Next 5 Years)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True)

plt.show()
