# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model
"""
import logging
import os

library_path = '/work/lib'

# 检查LD_LIBRARY_PATH是否已经存在
if 'LD_LIBRARY_PATH' in os.environ:
    # 如果已存在，将新的库路径添加到现有路径的末尾，使用':'作为分隔符
    os.environ['LD_LIBRARY_PATH'] = library_path + ':' + os.environ['LD_LIBRARY_PATH']
else:
    # 如果不存在，直接将新的库路径赋值给LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = library_path

value = os.environ.get('LD_LIBRARY_PATH')
print(value)
#import torch_mlu
import sys
sys.path.insert(0, "/work/yolov5")
sys.path.insert(0, "../yolov5")

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
from models.experimental import attempt_load
from utils.general import  non_max_suppression, scale_coords
from utils.augmentations import letterbox
import argparse
import cv2 
import torchvision
from torchvision.transforms import transforms
import numpy as np
import torch
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
       

    return output
import random
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img
def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
try:
    import torch_mlu.core.mlu_model as ct
    import torch_mlu.core.mlu_quantize as mlu_quantize
except:
    print('\033[0;31mimport torch_mlu failed in {}!!!\033[0m'.format(__file__))
from models.yolo import Model
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

from models.common import DetectMultiBackend, AutoShape

stats = "LOADING"

modeluri = ""

if MODEL_URI.endswith(".pt") or MODEL_URI.endswith(".onnx") or MODEL_URI.endswith(".torchscript"):
    modeluri = MODEL_URI
else:
    if os.path.exists(MODEL_URI + "/best.pt"):
        modeluri = MODEL_URI + "/best.pt"
    elif os.path.exists(MODEL_URI + "/last.pt"):
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
    # if opdevice == 'mlu':
    #     ct.set_cnml_enabled(False)


    # if opt.device != 'cpu' and opt.device != 'mlu':
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda:0')
    #     else:
    #         device = torch.device('cpu')
    # else:
    device = torch.device("cpu")
    with open(MODEL_URI+"/class_name.txt","r")as file:
        lines = file.readlines()

    nc = len(lines)
    names = [str(i) for i in range(nc)]

    model = Model("/work/yolov5/models/yolov5s.yaml", ch=3, nc=nc).eval()#.to(device)  # create
    if device.type == 'mlu':
        model = mlu_quantize.adaptive_quantize(model=model, steps_per_epoch=2, bitwidth=16)
    ckpt = torch.load(modeluri)

    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
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
    # for imdata in data:
    #     s = imdata["image_b64"].encode()
    #     s = s.translate(_urlsafe_decode_translation)
    #     mod = len(s) % 4
    #     if mod > 0:
    #         s += b'='*(4-mod)
    #     imagebytes = base64.decodebytes(s)
    #     data_stream = io.BytesIO(imagebytes)
    #     im = Image.open(data_stream)
    #     imgs.append(im)
    # results = model.forward(imgs)
    # ret = results.pandas().xyxy[0].to_dict(orient="records")
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
        img0 = img
        img = letterbox(img,  auto=False, stride=32)[0]
        img1 = img
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred, _ = model(im)


        if device.type == 'mlu':
            pred = pred.cpu()
        with open(MODEL_URI+"/class_name.txt","r")as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
        pred = non_max_suppression(pred, 0.3, 0.4, None, True)
        for i, det in enumerate(pred):
            import json
            print(det)
            data_list = []
            if len(det):
                # Rescale boxes from img_size to im0 sizes
                print(img0.shape)
                det[:, :4] = scale_boxes(img1.shape, det[:, :4], img0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    img_result = plot_one_box(xyxy, img0, line_thickness=3)
                    data_list_a =  {
            "class": int(cls.item()),  # 目标分类id
            "confidence": float(conf.item()),  # 目标置信度
            "name": lines[int(cls.item())],  # 目标分类名称
            "xmin": float(xyxy[0].item()),  # 目标位置矩形框左上角顶点X轴坐标值
            "ymin": float(xyxy[1].item()),  # 目标位置矩形框左上角顶点Y轴坐标值
            "xmax": float(xyxy[2].item()),  # 目标位置矩形框右下角顶点X轴坐标值
            "ymax": float(xyxy[3].item())  # 目标位置矩形框右下角顶点Y轴坐标值
        }
                    data_list.append(data_list_a)
                    print("xyxy:", xyxy)
                    print("conf  or cls",conf,cls)
    return jsonify({'outputs': data_list})

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
