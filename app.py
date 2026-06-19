from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form inputs
        kms_driven = int(request.form['kms_driven'])
        owner = int(request.form['owner'])
        car_age = int(request.form['car_age'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']

        # Initialize input dictionary
        input_dict = {
            'Kms_Driven': kms_driven,
            'Owner': owner,
            'Car_Age': car_age,
            'Fuel_Type_Diesel': 0,
            'Fuel_Type_Electric': 0,
            'Fuel_Type_LPG': 0,
            'Fuel_Type_Petrol': 0,
            'Seller_Type_Individual': 0,
            'Seller_Type_Trustmark Dealer': 0,
            'Transmission_Manual': 0
        }

        # Handle fuel type
        fuel_key = f'Fuel_Type_{fuel_type}'
        if fuel_key in input_dict:
            input_dict[fuel_key] = 1

        # Handle seller type
        seller_key = f'Seller_Type_{seller_type}'
        if seller_key in input_dict:
            input_dict[seller_key] = 1

        # Handle transmission
        if transmission == 'Manual':
            input_dict['Transmission_Manual'] = 1

        # Create input vector
        input_vector = np.array([input_dict[col] for col in input_dict.keys()]).reshape(1, -1)

        # Predict
        prediction = model.predict(input_vector)[0]
        prediction = f"{round(prediction):,}"

        return render_template('result.html',
                       prediction=prediction,
                               kms_driven=kms_driven,
                               owner=owner,
                               car_age=car_age,
                               fuel_type=fuel_type,
                               seller_type=seller_type,
                               transmission=transmission)


    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)