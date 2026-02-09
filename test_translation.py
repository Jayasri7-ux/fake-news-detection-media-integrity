import requests
import json

url = "http://127.0.0.1:5000/predict"
# English news text
eng_text = "The government has announced a new policy to support renewable energy projects across the country."

payload = {
    "text": eng_text,
    "is_url": False,
    "target_lang": "te" # User has Telugu selected in UI
}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Prediction: {result.get('prediction')}")
    print(f"Extracted/Display Text (Telugu): {result.get('extracted_text')}")
except Exception as e:
    print(f"Error: {e}")
