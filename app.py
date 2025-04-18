from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model with 5 input features
model = joblib.load("model/travel_rf_model.pkl")

# Mapping for categorical features
season_map = {'Summer': 0, 'Winter': 1, 'Monsoon': 2, 'Autumn': 3}
demand_map = {'Low': 0, 'Medium': 1, 'High': 2}
mode_map = {'Private Car': 0, 'Bus': 1, 'Train': 2, 'Flight': 3}
fuel_map = {'Petrol': 0, 'Diesel': 1, 'Electric': 2}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Get form data
        km = float(request.form['kilometers'])
        season = request.form['season']
        demand = request.form['demand']
        mode_of_transport = request.form['mode_of_transport']
        fuel_type = request.form['fuel_type']

        # Create feature array
        features = np.array([[km,
                              season_map[season],
                              demand_map[demand],
                              mode_map[mode_of_transport],
                              fuel_map[fuel_type]]])

        # Predict the cost
        predicted_price = model.predict(features)[0]

        # Generate dynamic explanation
        explanation = []

        if km > 500:
            explanation.append("Long distance increases cost significantly.")
        else:
            explanation.append("Shorter distance helps keep cost lower.")

        if demand == 'High':
            explanation.append("High demand typically raises travel prices.")
        elif demand == 'Low':
            explanation.append("Low demand often leads to cheaper travel.")

        if season == 'Winter':
            explanation.append("Winter is peak season, which can increase cost.")
        elif season == 'Monsoon':
            explanation.append("Monsoon might lower prices due to less travel.")

        if mode_of_transport == 'Flight':
            explanation.append("Flights are the most expensive travel mode.")
        elif mode_of_transport == 'Train':
            explanation.append("Trains offer a more economical option.")

        if fuel_type == 'Electric':
            explanation.append("Electric vehicles are cost-effective.")
        elif fuel_type == 'Petrol':
            explanation.append("Petrol costs moderately affect the total price.")

        detailed_explanation = " ".join(explanation)

        return render_template('result.html',
                               price=round(predicted_price, 2),
                               km=km,
                               season=season,
                               demand=demand,
                               mode_of_transport=mode_of_transport,
                               fuel_type=fuel_type,
                               explanation=detailed_explanation)

if __name__ == "__main__":
    app.run(debug=True)
