import json
import time
import base64
# import threading
import urllib
import urllib.request
from multiprocessing import Pool

with open('./unit_test/test_img/detection_test.jpg', 'rb') as f:
    base64_data_detection = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/register_test.jpg', 'rb') as f:
    base64_data_register = base64.b64encode(f.read()).decode()


def req_func(url, json_data, json_length):
    request = urllib.request.Request(url=url)
    request.add_header("Content-Type", "application/json")
    request.add_header("Content-Length", json_length)
    res = urllib.request.urlopen(url=request, data=json_data)
    result = json.load(res)
    return result


url = 'http://127.0.0.1:80/face_register'
register_input_dict = {"recordID": "test-20250424test",
                       "sessionID": "test-HJ5Loo5546xxxxxx",
                       "imgData": base64_data_register,
                       "msgID": "test-hshgsuhguhbjh2356vjijh",
                       "imgType": "JPG"}

json_data = json.dumps(register_input_dict).encode('utf-8')
json_length = len(json_data)

Pool_Num = 4
result = []
delta_time, delta_time_best = 0, 100000000
time_cons = {}

for Pn in range(2, Pool_Num, 1):
    time_start = time.time()
    p = Pool(Pn)   # 使用线程池建立子进程
    for i in range(Pn):   # 开始子进程
        result.append(p.apply_async(req_func, args=(url, json_data, json_length)))
    p.close()
    p.join()
    time_end = time.time()
    delta_time = ((time_end - time_start) * 1000) / Pn
    time_cons[Pn] = delta_time

    if delta_time < delta_time_best:
        delta_time_best = delta_time
        print('best pool num is {}, time consume is {} ms'.format(Pn, delta_time))
    print('face_register finished with {} pools in {} ms!.'.format(Pn, delta_time))

for i in result:
    rr = i.get()
    if 'faceFeature' in rr:
        faceFeature = rr['faceFeature']

# 异步非阻塞式多进程并行
time_start = time.time()
p = Pool(Pool_Num)   # 使用线程池建立子进程
for i in range(Pool_Num):   # 开始子进程
    result.append(p.apply_async(req_func, args=(url, json_data, json_length)))
p.close()
p.join()
time_end = time.time()
print('face_register finished with {} pools in {} ms!.'.format(Pool_Num, ((time_end - time_start) * 1000) / Pool_Num))

url = 'http://127.0.0.1:80/head_detection'
detection_input_dict = {"recordID": "test-20250424test",
                        "sessionID": "test-HJ5Loo5546xxxxxx",
                        "imgData": base64_data_detection,
                        "faceFeature": faceFeature,
                        "msgID": "test-hshgsuhguhbjh2356vjijh",
                        "imgType": "JPG"}

json_data = json.dumps(detection_input_dict).encode('utf-8')
json_length = len(json_data)
print('img ready again')

# 异步非阻塞式多进程并行
time_start = time.time()
p = Pool(Pool_Num)   # 使用线程池建立子进程
for i in range(Pool_Num):   # 开始子进程
    p.apply_async(req_func, args=(url, json_data, json_length))
p.close()
p.join()
time_end = time.time()
print('head_detection finished with {} pools in {} ms!'.format(Pool_Num, ((time_end - time_start) * 1000) / Pool_Num))
