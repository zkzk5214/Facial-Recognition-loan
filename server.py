# -*- coding: utf-8 -*-
import os
import re
import cv2
import json
import time
import base64
import traceback
import numpy as np
import yaml
import socket
from werkzeug.serving import WSGIRequestHandler
from flask import Flask, request, jsonify
import utils.log as logging

from inferencer.face_pipeline import face_register_, get_topk, head_detection_, StatusCode
from inferencer.dark_bg_detector import dark_bg_check

WSGIRequestHandler.protocol_version = 'HTTP/1.1'

hostname = socket.gethostname()
safe_hostname = re.sub(r'[^a-zA-Z0-9\-]', '_', hostname)

with open('./config.yaml', 'r') as f:
    _config = yaml.safe_load(f)
env = os.environ.get('APP_ENV', 'pro')
LOG_FILENAME = _config['log_path'][env].format(hostname=safe_hostname)

try:
    logger = logging.create_logger(LOG_FILENAME, "server_logging")
except Exception:
    logger = logging.create_logger("./log/server.log", "server_logging")

app = Flask(__name__)
app_start_version = str(time.ctime())

with open('./version', 'r') as f:
    code_version = f.read().strip()

try:
    server_ip = str(socket.gethostbyname(socket.gethostname()))
except socket.gaierror:
    server_ip = 'localhost'
version_description = f'[IP]: {server_ip}: [APP]: {app_start_version}: [CODE]: {code_version}\n'


def decode_base64_image(base64_data):
    return cv2.imdecode(np.frombuffer(base64.b64decode(base64_data), np.uint8), cv2.IMREAD_COLOR)


def process_common_request(request, img_data_key='imgData'):
    json_data = json.loads(request.get_data(as_text=True))
    img = decode_base64_image(json_data[img_data_key])
    return json_data, img


def create_response(json_data, default_resp_code=StatusCode.SUCCESS.value):
    return {
        "recordID": json_data['recordID'],
        "sessionID": json_data['sessionID'],
        "msgID": json_data['msgID'],
        'resp_code': default_resp_code
    }


def log_request_result(json_data, resp_code, step, additional_info=None):
    log_entry = ['[Return]:', str(step), str(json_data['recordID']), str(json_data['sessionID']),
                 str(json_data['msgID']), str(resp_code)]
    if additional_info:
        log_entry.append(str(additional_info))
    logger.info('\t'.join(log_entry) + '#\n')


def log_warning(tag, endpoint, json_data, resp_code, img_data=None):
    entry = [f'[{tag}]: {endpoint}',
             str(json_data['recordID']), str(json_data['sessionID']),
             str(json_data['msgID']), str(resp_code)]
    if img_data:
        entry.append(str(img_data))
    logger.info('\t'.join(entry) + '#\n')


@app.route('/face_register', methods=['POST'])
def face_register():
    try:
        start_time = time.time()
        json_data, img = process_common_request(request)
        res = create_response(json_data)
        face_features, status_code, quality_score = face_register_(img)

        res['imgQualityScore'] = quality_score
        res['resp_code'] = status_code

        if not face_features:
            log_warning('Warning-NoFeature', '/face_register', json_data,
                        res['resp_code'], json_data['imgData'])
            return jsonify(res)

        res['faceFeature'] = str(np.array(face_features[0]).tolist())

        latency_ms = str(int((time.time() - start_time) * 1000))
        log_request_result(json_data, res['resp_code'], '/face_register',
                           {'QualityScore': quality_score, 'TimeCost': latency_ms})
        return jsonify(res)

    except Exception as e:
        logger.error("[Traceback]:" + str(traceback.format_exc()).replace('\n', ' \t'))
        logger.error('\t'.join(['[ExceptionTriggered]: /face_register', str(e), '#\n']))
        return jsonify({'resp_code': StatusCode.ERROR.value, 'error_msg': str(e)})


@app.route('/head_detection', methods=['POST'])
def head_detection():
    try:
        start_time = time.time()
        json_data, img = process_common_request(request)
        res = create_response(json_data)
        res.update({"detectRes": 0, "top3Similarity": [], "faceSimilarity": [-1], "msg": ''})

        if 'faceFeature' not in json_data or len(json_data['faceFeature']) <= 10:
            res['resp_code'] = StatusCode.NO_FEATURE.value
            log_warning('Warning-NoFeature', '/head_detection', json_data, res['resp_code'])
            return jsonify(res)

        face_count, face_conf, face_features, status_code = head_detection_(img)

        if face_count < 1:
            res['resp_code'] = status_code
            res['msg'] = '看不到您的脸'
            log_warning('Warning-NoFace', '/head_detection', json_data, res['resp_code'])
            return jsonify(res)

        res['detectRes'] = face_count
        res['top3Similarity'] = face_conf
        res['faceSimilarity'] = [0]
        res['resp_code'] = status_code

        if status_code == StatusCode.BLURRED.value:
            res['msg'] = '不要晃动手机'
            log_warning('Warning-Blur', '/head_detection', json_data,
                        res['resp_code'], json_data['imgData'])
        elif status_code == StatusCode.BACKLIGHT.value:
            res['msg'] = '摄像头不要正对灯光'
            log_warning('Warning-Light', '/head_detection', json_data,
                        res['resp_code'], json_data['imgData'])

        if face_features is None or len(face_features) == 0:
            res['resp_code'] = StatusCode.FEATURE_EXTRACT_ERROR.value
            res['msg'] = '看不到您的脸'
            log_warning('Warning-Other', '/head_detection', json_data,
                        res['resp_code'], json_data['imgData'])
            return jsonify(res)

        register_vector = np.array([json.loads(json_data['faceFeature'])])
        max_score, _, vector = get_topk(register_vector, np.array(face_features))
        res['faceSimilarity'] = max_score if max_score[0] >= 0 else [0]

        latency_ms = str(int((time.time() - start_time) * 1000))
        log_request_result(json_data, res['resp_code'], '/head_detection',
                           {'Face_Count': res['detectRes'],
                            'Face_Similarity': res['faceSimilarity'],
                            'Face_Conf': res['top3Similarity'],
                            'TimeCost': latency_ms})

        return jsonify(res)

    except Exception as e:
        logger.error("[Traceback]: " + str(traceback.format_exc()).replace('\n', '\t'))
        logger.error('\t'.join(['[ExceptionTriggered]: /head_detection', str(e), '#\n']))
        return jsonify({'resp_code': StatusCode.ERROR.value, 'error_msg': str(e)})


@app.route('/dark_bg_check', methods=['POST'])
def dark_bg_check_endpoint():
    try:
        start_time = time.time()
        json_data, img = process_common_request(request)
        res = create_response(json_data)

        is_dark_bg, resp_code, dark_score, bg_ratio = dark_bg_check(img)
        res['resp_code'] = resp_code
        res['is_dark_bg'] = is_dark_bg
        res['dark_score'] = round(dark_score, 4)
        res['bg_ratio'] = round(bg_ratio, 4)

        latency_ms = str(int((time.time() - start_time) * 1000))
        log_request_result(json_data, res['resp_code'], '/dark_bg_check',
                           {'is_dark_bg': res['is_dark_bg'], 'dark_score': res['dark_score'],
                            'bg_ratio': res['bg_ratio'], 'TimeCost': latency_ms})
        return jsonify(res)

    except Exception as e:
        logger.error("[Traceback]:" + str(traceback.format_exc()).replace('\n', ' \t'))
        logger.error('\t'.join(['[ExceptionTriggered]: /dark_bg_check', str(e), '#\n']))
        return jsonify({'resp_code': StatusCode.ERROR.value, 'error_msg': str(e)})


@app.route('/version/', methods=['GET'])
def query_version():
    gunicorn_process_count = os.popen("pgrep -c gunicorn").read().strip()
    return f'{version_description}: [gunicorn]: {gunicorn_process_count}'


if __name__ == '__main__':
    app.run('0.0.0.0', port=80, threaded=False)
