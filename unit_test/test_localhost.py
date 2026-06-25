import sys
import json
import time
import base64
import urllib
import urllib.request
from multiprocessing import Pool


def req_func(url, json_data, json_length):
    request = urllib.request.Request(url=url)
    request.add_header("Content-Type", "application/json")
    request.add_header("Content-Length", json_length)
    res = urllib.request.urlopen(url=request, data=json_data)
    result = json.load(res)
    return result


def smoke_test(base):
    url = base + '/face_register'
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

    url = base + '/head_detection'
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


def load_test(base):
    url = base + '/face_register'
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

    faceFeature = None
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

    url = base + '/head_detection'
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


with open('./unit_test/test_img/detection_test.jpg', 'rb') as f:
    base64_data_detection = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/register_test.jpg', 'rb') as f:
    base64_data_register = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/black_bg_test.jpg', 'rb') as f:
    base64_data_black_bg = base64.b64encode(f.read()).decode()


def dark_bg_check_test(base):
    url = base + '/dark_bg_check'
    input_dict = {"recordID": "test-20250424test",
                  "sessionID": "test-HJ5Loo5546xxxxxx",
                  "imgData": base64_data_black_bg,
                  "msgID": "test-hshgsuhguhbjh2356vjijh",
                  "imgType": "JPG"}
    json_data = json.dumps(input_dict).encode('utf-8')
    json_length = len(json_data)
    request = urllib.request.Request(url=url)
    request.add_header("Content-Type", "application/json")
    request.add_header("Content-Length", json_length)
    res = urllib.request.urlopen(url=request, data=json_data)
    result = json.load(res)
    print('dark_bg_check:', result.get('is_dark_bg'),
          result.get('dark_score'), result.get('bg_ratio'), result.get('resp_code'))


if __name__ == '__main__':
    env = sys.argv[1] if len(sys.argv) > 1 else 'pro'
    port = 37709 if env == 'dev' else 80
    base = 'http://127.0.0.1:{}'.format(port)

    if env == 'pro':
        smoke_test(base)
    else:
        load_test(base)

    dark_bg_check_test(base)
