from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


app = Flask(__name__)

# ===========================
# 1️⃣ Load your trained model
# ===========================
# Make sure your model file (e.g. model.pkl) is in the same folder as app.py
model = joblib.load('crop_model.pkl')


# ====================================
# 2️⃣ Route for serving the main page
# ====================================
@app.route('/')
def home():
    return render_template('index.html')  # make sure your frontend file is inside 'templates' folder

# ===========================
# 3️⃣ Prediction API endpoint
# ===========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract values from request
        arr = np.array([[ 
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])

        # Model prediction
        prediction = model.predict(arr)[0]

        # Return result as JSON
        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

# ===========================
# 4️⃣ Run the app
# ===========================
if __name__ == '__main__':
    app.run(debug=True)
