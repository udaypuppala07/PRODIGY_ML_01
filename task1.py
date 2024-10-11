# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from a CSV file (ensure the file is in the same directory or provide the full path)
# Example: 'housing_data.csv' should contain columns: SquareFootage, Bedrooms, Bathrooms, Price
df = pd.read_csv('housing_data.csv')

# Features and target
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]  # Features
y = df['Price']  # Target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict house prices on the test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Get user input for the example house
square_footage = float(input("Enter the square footage of the house: "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))

# Create a DataFrame for the example house with the user inputs
example_house = pd.DataFrame([[square_footage, bedrooms, bathrooms]], columns=['SquareFootage', 'Bedrooms', 'Bathrooms'])

# Predict the price of the example house
predicted_price = model.predict(example_house)

# Output the predicted price
print(f"Predicted price for the house: ${predicted_price[0]:.2f}")
