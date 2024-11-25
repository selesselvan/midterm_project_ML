import requests

url = "http://localhost:9696/predict"

data = {
    "Gender": 1,
    "Age": 27,
    "Occupation": 9,
    "Sleep_Duration": 6.1,
    "Quality_of_Sleep": 6,
    "Physical_Activity_Level": 42,
    "Stress_Level": 6,
    "BMI_Category": 1,
    "Blood_Pressure": 22,
    "Heart_Rate": 77,
    "Daily_Steps": 4200
}


response = requests.post(url, json=data).json()
print(response)