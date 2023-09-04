import shutil
import sys

sys.path.insert(0, "/work/yolov5")
sys.path.insert(0, "../yolov5")
import os
from pathlib import Path
import random

'''
enviroments:
    WORKFLOW_NAME: workid
    DATA_PATH: datasetpath. split by ":". 
                each folders support:
                    1 flat imgs and txts
                    2 train,test,val folders inside
    DATA_SPLIT_RATIO: when folders contains flat imgs and txts, it split files with such params.example: [7:2:1]
    CLASS_NUM: class num of the databases. if not specified CLASS_LABELS, labels will be automatically produce.
    CLASS_LABELS: class labels, split by ":". if specified CLASS_LABELS, class num will be automatically calculate.
    IMAGE_SIZE: input size. default: 320
    PRETRAIN: pretrain model. default:yolov5s.pt . can use yolov5n,yolov5s,yolov5m,yolov5l,yolov5x
'''

FILE = Path(__file__).resolve()
ROOT = os.getcwd()

TASK_ID="workid/_tt"
if "WORKFLOW_NAME" in os.environ:
    TASK_ID = "workid/" + os.environ["WORKFLOW_NAME"]
outputdir = TASK_ID+'/output/'
if not os.path.isdir(outputdir):
        os.makedirs(outputdir)



def main():
    print("workid:", TASK_ID)
    if not os.path.isdir(TASK_ID):
        os.makedirs(TASK_ID)
        
    # if "DATA_PATH" in os.environ:
    #     dataset_pathes = os.environ["DATA_PATH"].split(":")
    # else:
    #     datasets_path = os.environ["DATASETS_PATH"]
    #     dataset_pathes = datasets_path.split(":")
    

    # if "DATA_SPLIT_RATIO" in os.environ:
    #     data_split_ratio = os.environ["DATA_SPLIT_RATIO"]
    # else:
    #     data_split_ratio = "7:2:1"
    # train_ratio, val_ratio, test_ratio = calRatio(data_split_ratio)
    dataset_path = os.environ["DATASETS_PATH"]+"/data"
    print(dataset_path)
    
    class_num = 0
    labels = []
    if "CLASS_LABELS" in os.environ:
        labels = os.environ["CLASS_LABELS"].split(":")
        print(labels)
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
    
   
    max_epchos = int(os.environ["MAX_EPOCHS"])
    import subprocess
    print("==========================",labels)
    
    with open(outputdir+"/class_name.txt", 'w') as file:
        file.write('\n'.join(labels))

    label_name_str = " ".join(labels)
    # 执行Shell命令
    command = "python /work/yolov5/data_trans.py --dataset_path {}  --label_name {}  --save_model {}best.pth --save_dir {}".format(dataset_path,label_name_str,outputdir,outputdir)
    os.system(command)
    params = {
        "label_name":labels,
        "project": "./out/train",
        "noautoanchor": True,
        "optimizer": "Adam",
        "batch_size": 32,
        "epochs": 300,
        "save_model": outputdir+"/best.pt",
        "device":"mlu"
    }
    if "MAX_EPOCHS" in os.environ:
        epochs = int(os.environ["MAX_EPOCHS"])
        params["epochs"] = epochs
    import train
    train.run(**params)
    pass
    
if __name__ == "__main__":
    main()
