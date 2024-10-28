import warnings
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pmdarima import auto_arima


"""
Simulates the spread of a cyber threat (via Spam attacks) using the SIR model, with the infection rate adjusted based on 
the forecasted spam attack probability using an ARIMA model.

Functions:
    sir_model(y, t, beta, gamma): Defines the SIR model dynamics.
    
Workflow:
   
1. Defines the SIR model function `sir_model`:
    - Takes the current values for susceptible (S), infected (I), and recovered (R) populations.
    - Uses infection rate (beta) and recovery rate (gamma) to calculate the rate of change for each population.

2. Loads cyber threat data (`cyber_data.csv`) and filters data for selected countries.
    - Selected countries include: Republic of India, Islamic Republic of Pakistan, Democratic Socialist Republic of Sri Lanka, 
      Kingdom of Bhutan, Federal Democratic Republic of Nepal, Islamic Republic of Afghanistan.

3. For each country:
    - Extracts the spam attack probability data.
    - Uses `auto_arima` to automatically select the best ARIMA model to forecast the next spam probability.
    - Adjusts the infection rate (`beta`) in the SIR model based on the ARIMA forecast.
    - Solves the SIR model equations to get the time evolution of susceptible, infected, and recovered populations over 60 months.

4. Results are stored in the `results` dictionary, which contains the susceptible, infected, and recovered populations for each country.

5. Finally, the results are plotted:
    - For each country, it plots the probability of being infected over time.
    - The plot title, labels, and grid are set accordingly for visualization.

Parameters:
    data (pd.DataFrame): Cyber threat data, containing spam attack probabilities for different countries.
    nearby_countries (list): A list of selected countries for which the analysis is conducted.
    N (int): Total population size for the SIR model (default: 1000).
    I0 (int): Initial number of infected individuals (default: 1).
    R0 (int): Initial number of recovered individuals (default: 10).
    S0 (int): Initial number of susceptible individuals (calculated as N - I0 - R0).
    y0 (list): Initial values for susceptible (S0), infected (I0), and recovered (R0) individuals.
    gamma (float): Recovery rate in the SIR model (default: 0.1).
    time_points (np.array): Time points over 60 months for which the SIR model is solved.

Returns:
    A plot showing the infected probability over time for each selected country, adjusted using ARIMA-forecasted infection rates.
"""

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

data = pd.read_csv('cyber_data.csv')

N = 1000  
I0 = 1    
R0 = 10  
S0 = N - I0 - R0  
y0 = [S0, I0, R0] 
gamma = 0.1 
time_points = np.linspace(0, 60, 60) 

nearby_countries = ['Republic of India', 'Islamic Republic of Pakistan', 'Democratic Socialist Republic of Sri Lanka', 'Kingdom of Bhutan', 'Federal Democratic Republic of Nepal', 'Islamic Republic of Afghanistan']
nearby_data = data[data['Country'].isin(nearby_countries)]

results = {}

warnings.filterwarnings("ignore", category=FutureWarning)

for country in nearby_countries:
    country_spam_data = nearby_data[nearby_data['Country'] == country]['Spam'].dropna()

    country_spam_data = country_spam_data.reset_index(drop=True)
    country_spam_data = country_spam_data.dropna()

    if len(country_spam_data) > 10:
        model = auto_arima(country_spam_data, seasonal=False, stepwise=True, trace=True)

        forecast = model.predict(n_periods=1)

        print(f"Forecast for {country}: {forecast}, Type: {type(forecast)}")

        if forecast is not None and not forecast.empty:
            avg_forecast = forecast.iloc[0]  
            beta_adjusted = 0.3 * avg_forecast

            solution = odeint(sir_model, y0, time_points, args=(beta_adjusted, gamma))
            S, I, R = solution.T

            results[country] = {
                "Susceptible": S,
                "Infected": I,
                "Recovered": R
            }

plt.figure(figsize=(12, 8))

for country, res in results.items():
    infected_probability = res["Infected"] / N 
    plt.plot(time_points, infected_probability, label=f"Infected Probability ({country})")

plt.title("SIR Model Simulation of Cyber Threat Spread via Spam (ARIMA Adjusted)", fontsize=14)
plt.xlabel("Time (Months)", fontsize=12)
plt.ylabel("Infected Probability", fontsize=12) 
plt.ylim(0, 1)  
plt.legend(loc='best', fontsize=10)
plt.grid(True)

plt.show()