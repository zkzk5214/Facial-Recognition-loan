import json
# import time
import base64
import urllib
import urllib.request

with open('./unit_test/test_img/detection_test.jpg', 'rb') as f:
    base64_data_detection = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/register_test.jpg', 'rb') as f:
    base64_data_register = base64.b64encode(f.read()).decode()

url = 'http://127.0.0.1:80/face_register'
register_input_dict = {"recordID": "test-20250424test",
                       "sessionID": "test-HJ5Loo5546xxxxxx",
                       "imgData": base64_data_register,
                       "msgID": "test-hshgsuhguhbjh2356vjijh",
                       "imgType": "JPG"}

json_data = json.dumps(register_input_dict).encode('utf-8')
json_length = len(json_data)
request = urllib.request.Request(url=url)
request.add_header("Content-Type", "application/json")
request.add_header("Content-Length", json_length)
res = urllib.request.urlopen(url=request, data=json_data)
reg_res = json.load(res)
print(reg_res['imgQualityScore'], reg_res['resp_code'])
faceFeature = reg_res['faceFeature']

url = 'http://127.0.0.1:80/head_detection'
detection_input_dict = {"recordID": "test-20250424test",
                        "sessionID": "test-HJ5Loo5546xxxxxx",
                        "imgData": base64_data_detection,
                        "faceFeature": faceFeature,
                        "msgID": "test-hshgsuhguhbjh2356vjijh",
                        "imgType": "JPG"}

json_data = json.dumps(detection_input_dict).encode('utf-8')
json_length = len(json_data)
request = urllib.request.Request(url=url)
request.add_header("Content-Type", "application/json")
request.add_header("Content-Length", json_length)
res = urllib.request.urlopen(url=request, data=json_data)
print(res.read())
