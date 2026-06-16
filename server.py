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

from inferencer import mask_detect
from inferencer.face_pipeline import face_register, get_topk

WSGIRequestHandler.protocol_version = 'HTTP/1.1'

hostname = socket.gethostname()
safe_hostname = re.sub(r'[^a-zA-Z0-9\-]', '_', hostname)

with open('./config.yaml', 'r') as f:
    _config = yaml.safe_load(f)
LOG_FILENAME = _config['log_path'].format(hostname=safe_hostname)

try:
    logger = logging.create_logger(LOG_FILENAME, "server_logging")
except Exception:
    logger = logging.create_logger("./log/serverlanke.log", "server_logging")

app = Flask(__name__)
app_start_version = str(time.ctime())

with open('./version', 'r') as f:
    code_version = f.read().strip()

server_ip = str(socket.gethostbyname(socket.gethostname()))
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


@app.route('/head_detection', methods=['POST'])
def head_detection():
    try:
        start_time = time.time()
        json_data = request.get_data(as_text=True)
        json_data = json.loads(json_data)
        logger.info("[RECEIVE]: RecordID: " + json_data['recordID'] + " MsgID" + json_data['msgID'] + "\n")

        res = {"recordID": json_data['recordID'], "detectionSimilarity": [],
               "msgID": json_data['msgID'], "faceSimilarityDS": [], "faceSimilarityCS": [],
               "imgQuality": {"size": 1, "blur": 1, "bright": 1, "dark": 1}, "mask": 1}

        # Check Image Size
        if len(json_data['imgData']) < 1000:
            res["imgQuality"]["size"] = 0
            logger.info("ImageSizeError: " + str(res) + "\n")
            return jsonify(res)

        # Decode Image (DS, CS, Group)
        imgDataRegisterDS = cv2.imdecode(np.fromstring(base64.b64decode(json_data['imgDataRegisterDS']), np.uint8),
                                          cv2.IMREAD_COLOR)
        imgDataRegisterCS = cv2.imdecode(np.fromstring(base64.b64decode(json_data['imgDataRegisterCS']), np.uint8),
                                          cv2.IMREAD_COLOR)
        imgData = cv2.imdecode(np.fromstring(base64.b64decode(json_data['imgData']), np.uint8), cv2.IMREAD_COLOR)

        # Mask with 0
        res["mask"] = 0 if mask_detect.get_class(imgData) else 1

        ds_feature = face_register(imgDataRegisterDS, get_conf=False)
        cs_feature = face_register(imgDataRegisterCS, get_conf=False)
        group_feature, detectionConf = face_register(imgData)

        res["detectionSimilarity"] = detectionConf

        if len(group_feature) < 1:  # No face
            logger.info("[IncompleteFaceGroup]: " + str(res) + "\n")
            return jsonify(res)

        if len(ds_feature) == 1:
            res["faceSimilarityDS"] = get_topk(np.array(ds_feature), np.array(group_feature))

        if len(cs_feature) == 1:
            res["faceSimilarityCS"] = get_topk(np.array(cs_feature), np.array(group_feature))

        time_cost = str(int((time.time() - start_time) * 1000))
        logger.info("[RETURN]: " + str(res) + " TimeCost: {}ms".format(time_cost) + "\n")

        return jsonify(res)


@app.route('/ds_head_detection', methods=['POST'])
def ds_head_detection():
    try:
        start_time = time.time()
        json_data = request.get_data(as_text=True)
        json_data = json.loads(json_data)
        logger.info("[RECEIVE]: MsgID: " + json_data['msgID'] + "\n")

        res = {"detectionSimilarity": [], "msgID": json_data['msgID'], "faceSimilarityDS": [],
               "imgQuality": {'size': 1, 'blur': 1, 'bright': 1, 'dark': 1}}

        # Image too large
        if len(json_data['imgData']) < 1000:
            res["imgQuality"]["size"] = 0
            logger.info("[ImageSizeError]: " + str(res) + "\n")
            return jsonify(res)

        # Decode Image (DS, Group)
        imgDataRegisterDS = cv2.imdecode(np.fromstring(base64.b64decode(json_data['imgDataRegisterDS']), np.uint8),
                                          cv2.IMREAD_COLOR)
        imgData = cv2.imdecode(np.fromstring(base64.b64decode(json_data['imgData']), np.uint8), cv2.IMREAD_COLOR)

        ds_feature = face_register(imgDataRegisterDS, get_conf=False)
        img_feature, detectionConf = face_register(imgData)

        res["detectionSimilarity"] = detectionConf

        if len(img_feature) < 1:
            logger.info("[InCompleteFaceGroup]: " + str(res) + "\n")
            return jsonify(res)

        if len(ds_feature) < 1:
            logger.info("[InCompleteFaceDS]: " + str(res) + "\n")
            return jsonify(res)

        res['faceSimilarityDS'] = get_topk(np.array(ds_feature), np.array(img_feature))

        time_cost = str(int((time.time() - start_time) * 1000))
        logger.info("[RETURN]: " + str(res) + " TimeCost: {}ms".format(time_cost) + "\n")
        return jsonify(res)

    except Exception as e:
        json_data = request.get_data(as_text=True)
        json_data = json.loads(json_data)

        res = {"detectionSimilarity": [100],
               "msgID": json_data['msgID'], "faceSimilarityDS": [100],
               "imgQuality": {'size': 1, 'blur': 1, 'bright': 1, 'dark': 1}}  # "sessionID":json_data['sessionID'],
        
        logger.error("[Traceback]: " + str(traceback.format_exc()).replace('\n', '\t'))
        logger.error('\t'.join(['[ExceptionTriggered]: /ds_head_detection', str(e), '#\n']))
        return jsonify(res)


@app.route('/version/', methods=['GET'])
def query_version():
    gunicorn_process_count = re.sub(' {1,}', '', os.popen("pgrep gunicorn | wc").read()).split(' ')[1]
    return f'{version_description}: [gunicorn]: {gunicorn_process_count}'


if __name__ == '__main__':
    app.run('0.0.0.0', port=80, threaded=False)
