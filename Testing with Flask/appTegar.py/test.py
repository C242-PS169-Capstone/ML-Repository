import requests

# URL endpoint
url = "http://127.0.0.1:5000/predict"

# Data JSON
data = {
    "text": "Lately, I've been feeling like there's no escape from the constant pain and hopelessness. Every day feels heavier than the last, and the thought of continuing seems unbearable. I've tried talking to people, but it feels like no one truly understands the depth of my despair. I'm tired of pretending everything is okay when it's not."
}


# Header
headers = {"Content-Type": "application/json"}

# Kirim permintaan POST
response = requests.post(url, json=data, headers=headers)

# Cetak hasil
if response.status_code == 200:
    print("Prediction:", response.json()['prediction'])
    print("Confidence:", response.json()['confidence'])
else:
    print("Error:", response.json())
