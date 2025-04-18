import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 300

# Generate synthetic data
data = pd.DataFrame({
    'kilometers': np.random.uniform(10, 1000, size=n_samples),
    'season': np.random.choice(['Summer', 'Winter', 'Monsoon', 'Autumn'], size=n_samples),
    'demand_level': np.random.choice(['Low', 'Medium', 'High'], size=n_samples),
    'mode_of_transport': np.random.choice(['Private Car', 'Bus', 'Train', 'Flight'], size=n_samples),
    'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Electric'], size=n_samples)
})

# Mapping categorical variables to numeric
season_map = {'Summer': 0, 'Winter': 1, 'Monsoon': 2, 'Autumn': 3}
demand_map = {'Low': 0, 'Medium': 1, 'High': 2}
mode_map = {'Private Car': 0, 'Bus': 1, 'Train': 2, 'Flight': 3}
fuel_map = {'Petrol': 0, 'Diesel': 1, 'Electric': 2}

# Apply mappings
data['season'] = data['season'].map(season_map)
data['demand_level'] = data['demand_level'].map(demand_map)
data['mode_of_transport'] = data['mode_of_transport'].map(mode_map)
data['fuel_type'] = data['fuel_type'].map(fuel_map)

# Generate 'price' based on a simple model
data['price'] = (
    100 + 
    data['kilometers'] * np.random.uniform(5, 8) + 
    data['demand_level'] * 300 + 
    data['season'] * 150 +
    data['mode_of_transport'] * 200 + 
    data['fuel_type'] * 100 +
    np.random.normal(0, 200, size=n_samples)
)

# Features (independent variables) and target (dependent variable)
X = data[['kilometers', 'season', 'demand_level', 'mode_of_transport', 'fuel_type']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Print the Mean Squared Error
print(f'Mean Squared Error: {mse}')

# Save the trained model to disk
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/travel_rf_model.pkl")
print("Model trained and saved to model/travel_rf_model.pkl")
