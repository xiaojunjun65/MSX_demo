
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.yolo import SegmentationModel
import torch
ckpt = torch.load("/workspace/volume/model-x/yoloseg/yolov5/yolov5-master/yolov5s-seg.pt")

model = SegmentationModel("/workspace/volume/model-x/yoloseg/yolov5/yolov5-master/models/segment/yolov5s-seg.yaml", ch=3, nc=80)
names = model.module.names if hasattr(model, 'module') else model.names
state_dict = ckpt['model'].float().state_dict()
model.load_state_dict(state_dict, strict=False)
