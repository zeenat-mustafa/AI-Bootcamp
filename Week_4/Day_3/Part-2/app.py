
from flask import Flask, request, jsonify
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load metadata and the latest model version
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

latest_version = metadata['versions'][-1]
latest_model = joblib.load(metadata['details'][latest_version]['joblib_file'])

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify(metadata)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = latest_model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
