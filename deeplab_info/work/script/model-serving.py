
import logging
import os


#import torch_mlu
import sys

import argparse
import io
import os
import base64

import torch
from flask import Flask, request, jsonify
from flask import Blueprint
from PIL import Image

from libs.auth import get_user_from_request
import cv2 as cv
# import torch.backends.cudnn as cudnn
import numpy as np

import argparse
import cv2 
import torchvision
from torchvision.transforms import transforms
import numpy as np
import torch
import math
from copy import copy
import torch
import torch.nn as nn
import sys

sys.path.insert(0, "/work/deeplabv3-plus-pytorch-main")
sys.path.insert(0, "../deeplabv3-plus-pytorch-main")
from deeplab import DeeplabV3

app = Flask(__name__)
bp = Blueprint('api', __name__, url_prefix='/')

MONGO_URI = os.environ.get('MONGO_URI')
VERSION_ID = os.environ.get('VERSION_ID')
NGINX_LOCATION = os.environ.get('NGINX_LOCATION')
NEED_AUTH = os.environ.get('NEED_AUTH', False)
API_KEY = os.environ.get("API_KEY", "73-5db9ea071dc17")

FRAMEWORK = os.environ.get('SERVING_TYPE', "PYTORCH").upper()
MODEL_URI = os.environ.get('MODEL_URI')

# -------- mnist-----------
SERVING_HOST = os.environ.get('SERVING_HOST', 'localhost')
SERVING_PORT = os.environ.get('SERVING_PORT', 5500)

DETECTION_URL = "/%s/object-detection" % (NGINX_LOCATION)

print("model uri:", MODEL_URI, "det uri:", DETECTION_URL)

stats = "LOADING"

modeluri = ""

if MODEL_URI.endswith(".pth") or MODEL_URI.endswith(".onnx") or MODEL_URI.endswith(".torchscript"):
    modeluri = MODEL_URI
else:
    if os.path.exists(MODEL_URI + "/best_epoch_weights.pth"):
        modeluri = MODEL_URI + "/best_epoch_weights.pth"
    elif os.path.exists(MODEL_URI + "/last_epoch_weights.pth"):
        modeluri = MODEL_URI + "/last_epoch_weights.pth"
    else:
        fnames = os.listdir(MODEL_URI)
        for fn in fnames:
            if fn.endswith(".pt") or MODEL_URI.endswith(".onnx") or MODEL_URI.endswith(".torchscript"):
                modeluri = MODEL_URI + "/" + fn
                break

if modeluri == "":
    print("model dose not exsits!")
    stats = "ERROR"
else:
    print("model file:", modeluri)
    label_list = []
    # 读取文件并逐行处理
    with open(MODEL_URI+'/class_name.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
          label_list.append(line)
    deeplab = DeeplabV3(model_path=modeluri)
    stats = "AVAILABLE"

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if request.method != "POST":
        return

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        results = model.forward(im, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")

@bp.route("%s/test" % (NGINX_LOCATION), methods=['GET'])
@bp.route('test', methods=["GET"])
def test():
    logging.info('test....')
    return 'test ok.'

_urlsafe_decode_translation = bytes.maketrans(b'-_', b'+/')

@bp.route('%s/predict/submit-input' % (NGINX_LOCATION), methods=['POST'])
@bp.route('predict/submit-input', methods=['POST'])
def submit_input():
    """

    验证token
    获取输入数据
    校验输入数据格式
    preprocess the data ?

    输入数据提交给模型服务代理
    获取模型服务代理输出
    处理输出数据

    返回response

    update the call_num every each success call.
    {
        image2array: 1,
        instances: [{,…}]
            0: {,…}
                image_b64: "iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACV..."
    }
    :return:
    """
    # get current user id if the token if valid
    user_id = get_user_from_request(API_KEY, request, NEED_AUTH)

    # TODO: log the user id
    logging.info(f'USER: {user_id} calling')

    if "inputs" in request.json:
        data = request.json['inputs']
    elif "instances" in request.json:
        data = request.json['instances']

    # real prediction method with the model proxy
    # ret = model_proxy.predict(data)
    imgs = []
    bs = request.json['image2array']
    
    for imdata in data:
        s = imdata["image_b64"].encode()
        s = s.translate(_urlsafe_decode_translation)
        mod = len(s) % 4
        if mod > 0:
            s += b'='*(4-mod)
        imagebytes = base64.decodebytes(s)
        #decoded_image = base64.b64decode(imdata["image_b64"].encode())
        nparr = np.frombuffer(imagebytes, np.uint8)
        # 解码为 OpenCV 格式的图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        from PIL import Image
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        import numpy
        r_image = deeplab.detect_image(image_pil, count=False, name_classes=label_list)
        numpy_image = numpy.array(r_image)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv.imencode('.jpg', opencv_image)
        base64_str = base64.b64encode(buffer).decode()
    return {"result_img":base64_str}

import time
from flask import Response
@bp.route('%s/predict/upload' % (NGINX_LOCATION), methods=['POST'])
@bp.route('predict/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    file_name=""
    for f in files:
        t = time.time()
        file_name=str(int(round(t * 1000)))+".mp4"
        file_path = os.path.join("/work/script", file_name)
        f.save(file_path)
    return jsonify({"file_path":file_name})

@bp.route('%s/predict/check/<string:filepath>' % (NGINX_LOCATION), methods=['GET'])
@bp.route('predict/check/<string:filepath>', methods=['GET'])
def check(filepath):
    """
        文件下载
    :return:
    """
    # file_path = request.form.get('filepath')
    filepath="/work/script/" + filepath
    if os.path.exists(filepath):
        return {"msg":"ok"}
    return {"msg":"wait"}


@bp.route('%s/predict/download/<string:filepath>' % (NGINX_LOCATION), methods=['GET'])
@bp.route('predict/download/<string:filepath>', methods=['GET'])
def download(filepath):
    """
        文件下载
    :return:
    """
    # file_path = request.form.get('filepath')
    filepath="/work/script/" + filepath
    filename = os.path.basename(filepath)
    if os.path.exists(filepath):
        response = Response(file_iterator(filepath))
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers["Content-Disposition"] = 'attachment;filename="{}"'.format(filename)
        return response
    return {"msg":"wait"}

def file_iterator(file_path, chunk_size=512):
    """
        文件读取迭代器
    :param file_path:文件路径
    :param chunk_size: 每次读取流大小
    :return:
    """
    with open(file_path, 'rb') as target_file:
        while True:
            chunk = target_file.read(chunk_size)
            if chunk:
                yield chunk
            else:
                break
from threading import Thread

@bp.route('%s/predict/submit-mp4input' % (NGINX_LOCATION), methods=['POST'])
@bp.route('predict/submit-mp4input', methods=['POST'])
def submitVideoinput():
    user_id = get_user_from_request(API_KEY, request, NEED_AUTH)

    # TODO: log the user id
    logging.info(f'USER: {user_id} calling')

    video_path = request.json['videoPath']
    gModel = request.json['gModel']
    t=Thread(target=handle_video, args=(gModel, video_path))
    t.start()
    output_path = os.path.basename(video_path).replace('.mp4', '_out.avi')
    # 文件已经存在则删除
    if os.path.exists(output_path):
        os.remove(output_path)
    return {"file_path": output_path}, 200

def handle_video(gModel,video_path):
    try:
        video(gModel,video_path)
    except:
        print('----error')

def video(gModel, video_path):
    # global g_model
    # g_model = attempt_load(gModel, device=device)
    video_path = os.path.join("/work/script", video_path)
    # 通过cv中的类获取视频流操作对象cap
    cap = cv.VideoCapture(video_path)
    # 调用cv方法获取cap的视频帧（帧：每秒多少张图片）
    fps = cap.get(cv.CAP_PROP_FPS)
    print('fps:', fps)
    # 获取cap视频流的每帧大小
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print('size:', (width, height))
    output_path = os.path.basename(video_path).replace('.mp4', '_1_out.avi')
    output_path = os.path.join(os.path.dirname(video_path), os.path.basename(output_path))
    fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')  # 用于avi格式的生成
    if os.path.exists(output_path):
        os.remove(output_path)
    out_writer = cv.VideoWriter(output_path, fourcc, int(fps / 2), (int(width / 2), int(height / 2)), True)
    error_count = 0
    frame_count = 0
    out_fps = int(fps / 2)
    # 每秒处理张数
    handle_frame_out = out_fps * 2
    while cap.isOpened():
        is_success, frame = cap.read()
        if not is_success:
            error_count += 1
            print('error', error_count)
            if error_count > 1000:
                break
            else:
                continue
        frame_count += 1
        if frame_count % 2 != 1:
            continue
        if frame_count >= fps:
            frame_count = 0
            continue
        if frame_count > handle_frame_out:
            continue
        # img = letterbox(frame, 640, stride=32)[0]
        results = model.forward([frame])
        target_list = results.pandas().xyxy[0].to_dict(orient="records")
        if len(target_list) > 0:
            for item in target_list:
                # item['xmin'] = item['xmin'] * width / 640
                # item['xmax'] = item['xmax'] * width / 640
                # item['ymin'] = item['ymin'] * height / 640
                # item['ymax'] = item['ymax'] * height / 640
                cv.putText(frame, "{} - {}".format(item['name'], '%.2f' % item['confidence']), (int(item['xmin']), int(item['ymin']) - 6), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                cv.rectangle(frame, (int(item['xmin']), int(item['ymin'])), (int(item['xmax'])  , int(item['ymax']) ), (0, 0, 255), 2)
        try:
            out_writer.write(cv.resize(frame, (int(width / 2), int(height / 2))))
            print('draw target')
        except Exception as e:
            print(e)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    print('cap release')
    cap.release()
    try:
        print('video_path:', video_path)
        srcFile=output_path
        output_path = output_path.replace('_1_out.avi', '_out.avi')
        print('srcFile:', srcFile)
        print('output_path:', output_path)
        os.rename(srcFile,output_path)
    except Exception as e:
        print(e)


@bp.route('%s/status' % (NGINX_LOCATION), methods=['GET'])
@bp.route('status', methods=['GET'])
def get_status():
    # get current user id if the token if valid
    user_id = get_user_from_request(API_KEY, request, NEED_AUTH)

    # TODO: log the user id
    logging.info(f'USER: {user_id} call')

    # ret = model_proxy.model_status

    return jsonify({'state': stats})

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
#     parser.add_argument("--port", default=5000, type=int, help="port number")
#     opt = parser.parse_args()

#     # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
#     torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

#     model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
#     app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat

def main():
    app.register_blueprint(bp)
    app.run(host="0.0.0.0", port=SERVING_PORT)  # debug=True causes Restarting with stat
    pass

if __name__ == "__main__":
    main()
