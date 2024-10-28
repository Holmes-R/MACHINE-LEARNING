import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

"""
Linear Regression Model for Cyber Attack Prediction

This script applies a linear regression model to predict the 'On Demand Scan' attack type based on the 
'Local Infection' attack data from a dataset of cyber attacks. The model is trained using the `LinearRegression` 
class from scikit-learn, and the performance is evaluated using mean squared error (MSE). 

Modules:
    - pandas: For reading and manipulating the CSV data.
    - numpy: For numerical operations (though not explicitly used here).
    - matplotlib.pyplot: For plotting and visualizing the regression results.
    - sklearn.linear_model.LinearRegression: To fit a linear regression model.
    - sklearn.model_selection.train_test_split: To split the dataset into training and testing sets.
    - sklearn.metrics.mean_squared_error: To calculate the mean squared error for model evaluation.

Workflow:
    1. **Load Data**:
        - The dataset 'cyber_data.csv' is read, and rows with missing values are dropped.
    
    2. **Define Features and Target**:
        - 'Local Infection' is used as the independent variable (X), and 'On Demand Scan' is the dependent variable (y).
    
    3. **Train-Test Split**:
        - The dataset is split into 80% training and 20% testing data using `train_test_split`.
    
    4. **Train Linear Regression Model**:
        - The linear regression model is trained on the training data.
    
    5. **Model Prediction and Evaluation**:
        - Predictions are made on the test set.
        - The model performance is evaluated using mean squared error (MSE).
        - The model's coefficients and intercept are printed.
    
    6. **Plotting**:
        - A scatter plot of the actual data is created, and the regression line is plotted for the test set predictions.
    
    7. **Future Prediction**:
        - A future prediction is made for a new 'Local Infection' value (e.g., 30), and the predicted 'On Demand Scan' is printed.

Variables:
    - `data`: The dataset loaded from the CSV file.
    - `X`: Independent variable ('Local Infection').
    - `y`: Dependent variable ('On Demand Scan').
    - `X_train`, `X_test`, `y_train`, `y_test`: Split training and testing data for model training and evaluation.
    - `linear_reg_model`: The linear regression model object.
    - `y_pred`: Predictions made by the model on the test set.
    - `mse`: Mean squared error calculated between the actual and predicted values.
    - `coefficients`: The slope of the linear regression line.
    - `intercept`: The intercept of the linear regression line.
    - `future_local_infection`: A DataFrame for future 'Local Infection' values.
    - `future_on_demand_scan`: Predicted 'On Demand Scan' for the future 'Local Infection' values.

Outputs:
    - Mean squared error of the model.
    - Coefficients and intercept of the regression model.
    - Scatter plot of actual data and the regression line.
    - Predicted 'On Demand Scan' for a given 'Local Infection' value.
"""




data = pd.read_csv('cyber_data.csv')

data = data.dropna()

X = data[['Local Infection']]  
y = data['On Demand Scan']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

y_pred = linear_reg_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')


coefficients = linear_reg_model.coef_
intercept = linear_reg_model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')  

plt.xlabel('Local Infection')
plt.ylabel('On Demand Scan')
plt.title('Linear Regression: Local Infection vs. On Demand Scan')

plt.legend()

plt.show()  
future_local_infection = pd.DataFrame([[30]], columns=['Local Infection'])
future_on_demand_scan = linear_reg_model.predict(future_local_infection)

future_on_demand_scan_clamped = max(0, min(1, future_on_demand_scan[0]))

print(f"Predicted On Demand Scan for Local Infection : {future_on_demand_scan_clamped}")


