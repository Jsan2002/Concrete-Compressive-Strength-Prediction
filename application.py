from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__, static_folder='static')

# Load the model once when the app starts
model = joblib.load('XGBoost_Regressor_model.pkl')

# Define expected columns and their types
EXPECTED_COLUMNS = ['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag']
COLUMN_TYPES = {col: float for col in EXPECTED_COLUMNS}

@app.context_processor
def inject_now():
    return {'now': datetime.utcnow}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = request.form.to_dict()

            # Validate input
            if not all(col in input_data for col in EXPECTED_COLUMNS):
                return render_template('predict.html',
                                       prediction_text="Error: Missing input fields.",
                                       show_result=True)

            # Create DataFrame and convert types in one step
            df = pd.DataFrame([input_data])
            df = df.astype(COLUMN_TYPES)

            # Check for any NaN values after conversion
            if df.isnull().any().any():
                return render_template('predict.html',
                                       prediction_text="Error: Invalid input. Please ensure all fields contain numeric values.",
                                       show_result=True)

            # Make prediction
            prediction = model.predict(df)
            result = f"{prediction[0]:.2f}"

            return render_template('predict.html',
                                   prediction_text=f"The Concrete compressive strength is {result} MPa",
                                   show_result=True)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return render_template('predict.html',
                                   prediction_text="An error occurred while processing your request.",
                                   show_result=True)
    
    # If it's a GET request, just show the prediction form
    return render_template('predict.html', show_result=False)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        input_data = request.json

        # Validate input
        if not all(col in input_data for col in EXPECTED_COLUMNS):
            return jsonify({"error": "Missing input fields"}), 400

        # Create DataFrame and convert types in one step
        df = pd.DataFrame([input_data])
        df = df.astype(COLUMN_TYPES)

        # Check for any NaN values after conversion
        if df.isnull().any().any():
            return jsonify({"error": "Invalid input. Please ensure all fields contain numeric values"}), 400

        # Make prediction
        prediction = model.predict(df)
        result = float(prediction[0])

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feature_description')
def feature_description():
    return render_template('feature_description.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)