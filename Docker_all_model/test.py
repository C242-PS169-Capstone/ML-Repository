import requests

# Define the endpoint
url = 'http://127.0.0.1:8000/predict'

# Test cases
test_cases = [
    {
        "description": "Test for Model 1 only",
        "data": {"text": "I feel hopeless and think about ending my life", "model": "model1"},
    },
    {
        "description": "Test for Model 2 only",
        "data": {"text": "I feel scared, anxious, what can I do? And may", "model": "model2"},
    },
    {
        "description": "Test for both models",
        "data": {"text": "I feel hopeless ", "model": "both"},
    },
    {
        "description": "Invalid model choice",
        "data": {"text": "I feel scared, anxious, what can I do? And may", "model": "model3"},
    },
    {
        "description": "Empty text input",
        "data": {"text": "", "model": "model1"},
    },
]

# Perform the tests
for case in test_cases:
    print(f"Running: {case['description']}")
    response = requests.post(url, json=case['data'])
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Error Response:", response.json())
    print("-" * 50)
