# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model
"""
import logging
import os

library_path = '/work/lib'


value = os.environ.get('LD_LIBRARY_PATH')
print(value)
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

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)

class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1, downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class LEDNet(nn.Module):
    def __init__(self,blocks, in_channel, num_classes=2, expansion=4):
        super(LEDNet,self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=in_channel, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def LEDNet50(in_channel, num_classes):
    return LEDNet([3, 4, 6, 3], in_channel, num_classes)


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
    if os.path.exists(MODEL_URI + "/best.pth"):
        modeluri = MODEL_URI + "/best.pth"
    elif os.path.exists(MODEL_URI + "/last.pth"):
        modeluri = MODEL_URI + "/last.pt"
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
    label_set = set()
    # 读取文件并逐行处理
    with open(MODEL_URI+'/label.txt', 'r') as file:
        lines = file.readlines()
        lines.reverse() 
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                label = parts[1]
                label_set.add(label)
                print(label_set)

    label_list = list(label_set)
   
    print(label_list)

    orig_model =  LEDNet50(3, len(label_list))
    orig_model.load_state_dict(torch.load(modeluri)['model'],strict=False)
    orig_model.eval()
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
        from torchvision.transforms import transforms
        transforms = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
        
        preprocessed_image = transforms(image_pil).unsqueeze(0)
        print(preprocessed_image.size())
        output = orig_model(preprocessed_image)
        import torch.nn.functional as F
        softmax_probs =np.argmax(output.detach().numpy(), axis=1)[0]
        print("Softmax Probabilities:", softmax_probs)
        print(label_list)
        data11 = {"class":label_list[softmax_probs]}
    return data11

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
