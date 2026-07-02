import requests
import json

url = "http://127.0.0.1:8001/run"
payload = {
    "appName": "app",
    "userId": "test_user",
    "sessionId": "test_session",
    "newMessage": {
        "role": "user",
        "parts": [{"text": "Hello, who are you?"}]
    }
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print("Status Code:", response.status_code)
print("Response Text:")
print(response.text)
