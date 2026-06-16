import json
import base64
import requests

with open('./unit_test/test_img/1.jpg', 'rb') as f:
    base64_data_register_ds = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/2.jpg', 'rb') as f:
    base64_data_register_cs = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/5.jpg', 'rb') as f:
    base64_data_detection = base64.b64encode(f.read()).decode()

url = 'http://127.0.0.1:80/head_detection'

detection_input_dict = {"recordID": "test-20210821test",
    "sessionID": "test-HJ5Loo5546xxxxxx",
    "saleID": "test-HJ5Loo5546xxxxxx",
    "imgData": base64_data_detection,
    "imgDataRegisterDS": base64_data_register_ds,
    "imgDataRegisterCS": base64_data_register_cs,
    "msgID": "test-hshgsuhguhbjh2356vjijh",
    "imgType": "JPG"}

json_data = json.dumps(detection_input_dict).encode('utf-8')
json_length = len(json_data)

for i in range(10):
    request = requests.post(url=url, data=json_data)
print("head_detection:", request.text)

url = 'http://127.0.0.1:80/ds_head_detection'

detection_input_dict = {"recordID": "test-20210821test",
    "sessionID": "test-HJ5Loo5546xxxxxx",
    "imgData": base64_data_detection,
    "imgDataRegisterDS": base64_data_register_ds,
    "msgID": "test-hshgsuhguhbjh2356vjijh",
    "imgType": "JPG"}

json_data = json.dumps(detection_input_dict).encode('utf-8')
json_length = len(json_data)

for i in range(10):
    request = requests.post(url=url, data=json_data)
print("ds_head_detection:", request.text)
