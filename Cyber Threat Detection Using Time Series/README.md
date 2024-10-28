# Cyber Threat Detection Using Time Series 

## Objective :
	The goal is to predict and model how cyber threats may evolve over time and spread across different countries. The system seems to use real-world data with attack probabilities for various types of cyber threats in different countries.

## Overview :
  The project employs methods such as ARIMA (AutoRegressive Integrated Moving Average) for time series forecasting, linear regression for predictive modeling, and a SIR (Susceptible-Infected-Recovered) model to simulate the spread of cyber threats .
  
 ARIMA Forecasting: The project implements rolling forecasts for specific attack types  using ARIMA models. The dataset is split into training and testing sets, and the model is trained to predict future occurrences of attacks over a specified period 
 
  Performance Evaluation: The performance of the ARIMA model is assessed using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
  
 Linear Regression: A linear regression model is utilized to predict the occurrence of the 'On Demand Scan' attack based on the 'Local Infection' attack data. The model is trained on 80% of the data, and its predictions are evaluated on the remaining 20%. 
 
 Future Predictions: The model can also make predictions based on hypothetical future values of 'Local Infection'.
 
  SIR Model: The project simulates the spread of cyber threats using a modified SIR model, adjusted based on ARIMA forecasts for nearby countries. This involves calculating the probabilities of being infected by cyber threats over time, offering a dynamic view of potential threats.


