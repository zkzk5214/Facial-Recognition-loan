import sys
import json
import time
import base64
import requests
from multiprocessing import Pool


with open('./unit_test/test_img/1.jpg', 'rb') as f:
    img_ds = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/2.jpg', 'rb') as f:
    img_cs = base64.b64encode(f.read()).decode()

with open('./unit_test/test_img/5.jpg', 'rb') as f:
    img_group = base64.b64encode(f.read()).decode()

head_input_dict = {
    "recordID": "test-20210821test",
    "sessionID": "test-HJ5Loo5546xxxxxx",
    "saleID": "test-HJ5Loo5546xxxxxx",
    "imgData": img_group,
    "imgDataRegisterDS": img_ds,
    "imgDataRegisterCS": img_cs,
    "msgID": "test-hshgsuhguhbjh2356vjijh",
    "imgType": "JPG",
}

ds_input_dict = {
    "recordID": "test-20210821test",
    "sessionID": "test-HJ5Loo5546xxxxxx",
    "imgData": img_group,
    "imgDataRegisterDS": img_ds,
    "msgID": "test-hshgsuhguhbjh2356vjijh",
    "imgType": "JPG",
}


def req_func(url, json_data):
    resp = requests.post(url, data=json_data,
                         headers={"Content-Type": "application/json"})
    return resp.json()


def smoke_test(base):
    url = base + '/head_detection'
    json_data = json.dumps(head_input_dict).encode('utf-8')
    res = req_func(url, json_data)
    print("head_detection:", res)

    url = base + '/ds_head_detection'
    json_data = json.dumps(ds_input_dict).encode('utf-8')
    res = req_func(url, json_data)
    print("ds_head_detection:", res)


def load_test(base):
    pool_num = 4
    result = []
    delta_time, delta_time_best = 0, 100000000

    url = base + '/head_detection'
    json_data = json.dumps(head_input_dict).encode('utf-8')

    for pn in range(1, pool_num + 1):
        time_start = time.time()
        p = Pool(pn)
        for _ in range(pn):
            result.append(p.apply_async(req_func, args=(url, json_data)))
        p.close()
        p.join()
        time_end = time.time()
        delta_time = ((time_end - time_start) * 1000) / pn
        if delta_time < delta_time_best:
            delta_time_best = delta_time
        print(f'head_detection finished with {pn} pools in {delta_time:.0f} ms!')

    url = base + '/ds_head_detection'
    json_data = json.dumps(ds_input_dict).encode('utf-8')

    for pn in range(1, pool_num + 1):
        time_start = time.time()
        p = Pool(pn)
        for _ in range(pn):
            result.append(p.apply_async(req_func, args=(url, json_data)))
        p.close()
        p.join()
        time_end = time.time()
        delta_time = ((time_end - time_start) * 1000) / pn
        if delta_time < delta_time_best:
            delta_time_best = delta_time
        print(f'ds_head_detection finished with {pn} pools in {delta_time:.0f} ms!')

    last = result[-1].get() if result else None
    print(f'\nbest time: {delta_time_best:.0f} ms, last result: {last}')


if __name__ == '__main__':
    env = sys.argv[1] if len(sys.argv) > 1 else 'pro'
    port = 37701 if env == 'dev' else 80
    base = f'http://127.0.0.1:{port}'

    if env == 'pro':
        smoke_test(base)
    else:
        load_test(base)
