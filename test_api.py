import requests
import json


# url
url = "http://0.0.0.0:1211/api/triton/t5/"

# sentence
sentence = "this hotel is quite good"

# payload
payload = {'data': sentence}

# response
res = requests.post(url, json=payload)
res = json.loads(res.text)
print(res['output'])