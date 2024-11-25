from flask import Flask, request, jsonify
import pickle
import numpy as np

# Create the Flask application
app = Flask(__name__)

# Load the trained Random Forest model
model_filename = 'random_forest_model.pkl'  

with open(model_filename, 'rb') as file:
    rf_model = pickle.load(file)

# Define a mapping dictionary
sleep_disorder_mapping = {
    0: "Not Present",
    1: "Sleep Apnea",
    2: "Insomnia"
}    

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    input_features = [
        data['Gender'],
        data['Age'],
        data['Occupation'],
        data['Sleep_Duration'],
        data['Quality_of_Sleep'],
        data['Physical_Activity_Level'],
        data['Stress_Level'],
        data['BMI_Category'],
        data['Blood_Pressure'],
        data['Heart_Rate'],
        data['Daily_Steps']
    ]

    # Make prediction
    prediction = rf_model.predict([input_features])

    # Convert numerical prediction to string label
    predicted_label = sleep_disorder_mapping[prediction[0]]

    return jsonify({'prediction': predicted_label})

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
