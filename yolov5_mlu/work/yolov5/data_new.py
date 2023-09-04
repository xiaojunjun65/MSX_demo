import argparse
from utils.general import (Logging, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
import os
import shutil
import sys

sys.path.insert(0, "/work/yolov5")
sys.path.insert(0, "../yolov5")
import os
from pathlib import Path
import random

from all2yolo import all2yolo,convert
import random
import xml.etree.ElementTree as ET
FILE = Path(__file__).resolve()
ROOT = os.getcwd()

TASK_ID="workid/_tt"
if "WORKFLOW_NAME" in os.environ:
        TASK_ID = "workid/" + os.environ["WORKFLOW_NAME"]
if not os.path.isdir(TASK_ID):
        os.makedirs(TASK_ID)
        
if "DATA_PATH" in os.environ:
    dataset_pathes = os.environ["DATA_PATH"].split(":")
else:
    dataset_pathes= os.environ["DATASETS_PATH"]+'/data'
    #dataset_pathes = datasets_path.split(":")


#if "DATA_SPLIT_RATIO" in os.environ:
 #   data_split_ratio = os.environ["DATA_SPLIT_RATIO"]
#else:
 #   data_split_ratio = "7:2:1"
#train_ratio, val_ratio, test_ratio = calRatio(data_split_ratio)


#rint("process data path:", dataset_pathes)
#print("data_split_ratio:", data_split_ratio)

# prepare train/val/test folder

train_path=[]
val_path=[]
test_path=[]

plain_files = []
for dpath in dataset_pathes:
    if os.path.isdir(dpath):
        for spath in os.listdir(dpath):
            fullpath = os.path.join(dpath, spath)
            if spath == "train":
                train_path.append(fullpath)
            elif spath == "val":
                val_path.append(fullpath)
            elif spath == "test":
                test_path.append(fullpath)
            elif spath.endswith(".jpg") or spath.endswith(".png"):
                plain_files.append(fullpath)
    elif os.path.isfile(dpath):
        pass

if len(plain_files) > 10:
    random.shuffle(plain_files)
    
    train_split_idx = len(plain_files)*train_ratio
    val_split_idx = len(plain_files)*(train_ratio+val_ratio)
    
    trainf = open("%s/_temp_train.txt"%(TASK_ID), "w")
    valf = open("%s/_temp_val.txt"%(TASK_ID), "w")
    testf = open("%s/_temp_test.txt"%(TASK_ID), "w")
    for i, pp in enumerate(plain_files):
        if i < train_split_idx:
            trainf.write("%s\n" % (pp))
        elif i < val_split_idx:
            valf.write("%s\n" % (pp))
        else:
            testf.write("%s\n" % (pp))
            
    trainf.close()
    valf.close()
    testf.close()
    
    train_path.append("%s/_temp_train.txt"%(TASK_ID))
    val_path.append("%s/_temp_val.txt"%(TASK_ID))
    test_path.append("%s/_temp_test.txt"%(TASK_ID))

class_num = 0
labels = []
if "CLASS_LABELS" in os.environ:
    labels = os.environ["CLASS_LABELS"].split(":")
    class_num = len(labels)
elif "CLASS_NUM" in os.environ:
    class_num = int(os.environ["CLASS_NUM"])
    labels = ["label_%d"%i for i in range(class_num)]
else:
    # standard coco classes
    labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']
    class_num = len(labels)

# write file:
with open("%s/_temp_data.yaml" % TASK_ID, "w") as ywr:
    ywr.write("path: %s \n" % ROOT)
    ywr.write("train: \n")
    for pp in train_path:
        ywr.write(" - %s\n" %  pp)
    ywr.write("val: \n")
    for pp in val_path:
        ywr.write(" - %s\n" %  pp)
    ywr.write("test: \n")
    for pp in test_path:
        ywr.write(" - %s\n" %  pp)
    
    ywr.write("nc: %d\n" %  class_num)
    ywr.write("names: [%s]\n"  %  ", ".join(["'%s'" % ss for ss in labels]))

