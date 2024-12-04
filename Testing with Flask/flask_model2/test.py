import requests

url = 'http://127.0.0.1:5000/predict'
data = {"text": "My emotions swing wildly, and I often feel empty. One minute, I’m deeply in love, the next, I’m convinced people hate me. Relationships are a constant struggle, and I can’t seem to control the intense, unpredictable feelings that overwhelm me."}
response = requests.post(url, json=data)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed with status code:", response.status_code)
