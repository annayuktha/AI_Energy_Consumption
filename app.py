from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model and scaler
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World'


@app.route('/predict/', methods=['POST'])
# @app.route('/predict')
def predict():
    try:
        # Get JSON data
        print("Hello!")
        data = request.get_json()

        # Extract 'hour' and 'day' from input
        hour = data.get("hour")
        day = data.get("day")
        print("Day: ",day,"\nhour: ",hour)

        if hour is None or day is None:
            return jsonify({"error": "Missing 'hour' or 'day' in request"}), 400
        
        # One-hot encode day (assuming days are numbered 0-6)
        day_vector = [1 if i == day else 0 for i in range(7)]
        # print(type(day_vector))
        # print(day_vector)

        # Prepare input features
        input_features = np.array([hour] + day_vector).reshape(1, -1)
        # print(type(input_features))
        # print(input_features)

        # Scale input
        input_scaled = scaler.transform(input_features)
        print(type(input_scaled))
        print(input_scaled)
        # Predict energy consumption
        prediction = mlp_model.predict(input_scaled)[0]

        return jsonify({"predicted_energy_consumption": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == '__main__':
    app.run(debug=True)
