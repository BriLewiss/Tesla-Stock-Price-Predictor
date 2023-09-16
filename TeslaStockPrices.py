import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("~/Downloads/TSLA.csv")

# Convert the Date column to Pandas Tiemstamps
data['Date'] = pd.to_datetime(data['Date'])

# Define the dependent and independent variable
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train,y_train)

# Input a date for prediction
input_date = input('Enter date in the format YEAR-MONTH-DAY') 

# Find the closest available date in the dataset
date_index = data['Date'].sub(pd.to_datetime(input_date)).abs().idxmin()

# Extract the feature values (open, high, low, volume) associated with the closest date
input_features = X.iloc[[date_index]]

# Make a prediction
predicted_price = model.predict(input_features)[0]

print(f"Predicted Stock Price on {input_date}: {predicted_price}")
