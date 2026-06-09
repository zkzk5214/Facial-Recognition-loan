# -*- coding: utf-8 -*-
import os
import re
import cv2
import json
import time
import base64
import traceback
import numpy as np
from werkzeug.serving import WSGIRequestHandler    # Web Server Gateway Interface
WSGIRequestHandler.protocol_version = 'HTTP/1.1'    # http server
import socket
socket.setdefaulttimeout(1)  # Prevent crawling pages for too long
from flask import Flask, request, jsonify
import utils.log as logging

# SIT
hostname = socket.gethostname()
safe_hostname = re.sub(r'[^a-zA-Z0-9\-]', '_', hostname)
LOG_FILENAME = f'/mnt/pvc/pvc-ph-irisk-id9743-vol723550-stg/log/hf_xy/info_{safe_hostname}.log'

try:
    logger = logging.LoggerGenerator(LOG_FILENAME, 'server_logging').get_logger()
except Exception as e:
    print(f"logger init failed: {e}")


from inferencer.register import face_register_, get_topk, head_detection_

app = Flask(__name__)
app_start_version = str(time.ctime())
code_version = str(open('./version', 'r').read())
this_ip = str(socket.gethostbyname(socket.gethostname()))
version_description = f'[IP]: {this_ip}: [APP]: {app_start_version}: [CODE]: {code_version}\n'


def decode_base64_image(base64_data):
    return cv2.imdecode(np.fromstring(base64.b64decode(base64_data), np.uint8), cv2.IMREAD_COLOR)


def process_common_request(request, img_data_key='imgData'):
    json_data = json.loads(request.get_data(as_text=True))
    img = decode_base64_image(json_data[img_data_key])
    return json_data, img


def create_response(json_data, default_resp_code=100):
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


@app.route('/face_register', methods=['POST'])
def face_register():
    try:
        start_time = time.time()
        json_data, img = process_common_request(request)
        res = create_response(json_data)
        img_feature, resp_flag, imgQualityScore = face_register_(img)  # feature is []

        res['imgQualityScore'] = imgQualityScore
        res['resp_code'] = resp_flag   # 200 Blur 201 逆光 221 Blur+逆光 300 No Face 400 Multi-Face

        if not img_feature:
            logger.info('\t'.join(['[Warning-NoFeature]: /face_register',
                str(json_data['recordID']), str(json_data['sessionID']),
                str(json_data['msgID']),
                str(res['resp_code']), str(json_data['imgData'])]) + '#\n')
            return jsonify(res)

        res['faceFeature'] = str(np.array(img_feature[0]).tolist())

        time_cost = str(int((time.time() - start_time) * 1000))
        log_request_result(json_data, res['resp_code'], '/face_register',
                           {'QualityScore': imgQualityScore, 'TimeCost': time_cost})
        return jsonify(res)
    
    except Exception as e:
        logger.error("[Traceback]:" +str(traceback.format_exc()).replace('\n', ' \t'))
        logger.error('\t'.join(['[ExceptionTriggered]: /face_register', str(e), '#\n']))
        return jsonify({'resp_code': 999, 'error_msg': str(e)})


@app.route('/head_detection', methods=['POST'])
def head_detection():
    try:
        start_time = time.time()
        json_data, img = process_common_request(request)
        res = create_response(json_data)
        res.update({"detectRes": 0, "top3Similarity": [], "faceSimilarity": [-1], "msg": ''})

        if 'faceFeature' in json_data and len(json_data['faceFeature']) > 10:
            face_count, face_conf, img_feature, resp_flag = head_detection_(img)

            if face_count < 1: # No Face
                res['resp_code'] = resp_flag
                res['msg'] = '看不到您的脸'
                logger.info('\t'.join(['[Warning-NoFace]: /head_detection',
                                       str(json_data['recordID']), str(json_data['sessionID']),
                                       str(json_data['msgID']),
                                       str(res['resp_code'])]) + '#\n')
            else:
                res['detectRes'] = face_count
                res['top3Similarity'] = face_conf
                res['faceSimilarity'] = [0]
                res['resp_code'] = resp_flag

                if resp_flag == 200:
                    res['msg'] = '不要晃动手机'
                    logger.info('\t'.join(['[Warning-Blur]: /head_detection',
                                           str(json_data['recordID']), str(json_data['sessionID']),
                                           str(json_data['msgID']),
                                           str(res['resp_code']), str(json_data['imgData'])]) + '#\n')

                if resp_flag == 201:
                    res['msg'] = '摄像头不要正对灯光'
                    logger.info('\t'.join(['[Warning-Light]: /head_detection',
                                           str(json_data['recordID']), str(json_data['sessionID']),
                                           str(json_data['msgID']),
                                           str(res['resp_code']), str(json_data['imgData'])]) + '#\n')

                if type(img_feature) != type(None) and len(img_feature) > 0:
                    register_vector = np.array([eval(json_data['faceFeature'])])
                    max_score, _, vector = get_topk(register_vector, np.array(img_feature))
                    res['faceSimilarity'] = max_score if max_score[0] >= 0 else [0]

                    time_cost = str(int((time.time() - start_time) * 1000))
                    log_request_result(json_data, res['resp_code'], '/head_detection',
                                    {'Face_Count': res['detectRes'],
                                        'Face_Similarity': res['faceSimilarity'],
                                        'Face_Conf': res['top3Similarity'],
                                        'TimeCost': time_cost})
                else:  # img_feature = []
                    res['resp_code'] = 600
                    res['msg'] = '看不到您的脸'
                    logger.info('\t'.join(['[Warning-Other]: /head_detection',
                                        str(json_data['recordID']), str(json_data['sessionID']),
                                        str(json_data['msgID']), str(res['resp_code']),
                                        str(json_data['imgData'])]) + '#\n')

        else:  # No faceFeature/wrong faceFeature type
            res['resp_code'] = 500
            logger.info('\t'.join(['[Warning-NoFeature]: /head_detection',
                                str(json_data['recordID']), str(json_data['sessionID']),
                                str(json_data['msgID']), str(res['resp_code'])]) + '#\n')
        return jsonify(res)

    except Exception as e:
        logger.error("[Traceback]: " + str(traceback.print_exc()).replace('\n', '\t'))
        logger.error(
            '\t'.join(['[ExceptionTriggered]: /head_detection ', str(e), '#\n']))
        return jsonify({'resp_code': 999, 'error_msg': str(e)})


@app.route('/version/', methods=['GET'])
def query_version():
    gunicorn_process_count = re.sub(' {1,}', '', os.popen("pgrep gunicorn | wc").read()).split(' ')[1]  # workers
    return f'{version_description}: [gunicorn]: {gunicorn_process_count}'


if __name__ == '__main__':
    app.run('0.0.0.0', port=80, threaded=False)
    