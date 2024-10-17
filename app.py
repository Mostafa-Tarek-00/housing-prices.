from flask import Flask, request, render_template, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load saved model and dataset
model = joblib.load('best_xgb_model_house_price.pkl')
housing_data = pd.read_csv('housing_prices.csv')

# Preprocess the data (for scaling input features)
current_year = 2024
housing_data['House_Age'] = current_year - housing_data['Age']
X = housing_data.drop(['Price', 'Age', 'Location_Rating'], axis=1)
scaler = MinMaxScaler()
scaler.fit(X)  # Fit scaler on the training data

# Route for the homepage with input form
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        input_data = request.get_json()

        # Convert input values to appropriate types
        square_feet = float(input_data['Square_Feet'])  # Ensure float
        bedrooms = int(input_data['Bedrooms'])            # Ensure int
        age = int(input_data['Age'])                      # Ensure int
        location_rating = float(input_data['Location_Rating'])  # Ensure float

        # Feature engineering
        house_age = current_year - age  # Perform calculation

        # Prepare new input DataFrame
        X_new = pd.DataFrame({
            'Square_Feet': [square_feet],
            'Bedrooms': [bedrooms],
            'House_Age': [house_age]
        })

        # Scale input data using the fitted scaler
        X_new_scaled = scaler.transform(X_new)

        # Predict price using the loaded model
        predicted_price = model.predict(X_new_scaled)[0]

        # Convert predicted price to standard Python float
        predicted_price = float(predicted_price)  

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter valid numerical values.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Route for visualizing the dataset
@app.route('/visualize')
def visualize():
    plt.figure(figsize=(10, 6))
    plt.scatter(housing_data['Square_Feet'], housing_data['Price'], alpha=0.5)
    plt.xlabel('Square Feet')
    plt.ylabel('Price')
    plt.title('Square Feet vs Price')
    plt.savefig('static/plot.png')  # Save the plot

    return send_file('static/plot.png', mimetype='image/png')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
