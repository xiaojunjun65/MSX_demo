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

def calRatio(data_split_ratio : str):
    ratestrs = data_split_ratio.split(":")
    rates = [float(ss) for ss in ratestrs]
    sumf = sum(rates)
    ratios = [0.0 for _ in range(3)]
    for i, ss in enumerate(rates):
        if i >= len(ratios):
            break
        ratios[i] = ss/sumf
    
    return ratios

def prepareData():
    '''
    scan dataset path. make _train_temp.txt, _test_temp.txt, _val_temp.txt
    ------------------------
    _temp_data.yaml:
    ------------------------
    path: /
    train: # train images (relative to 'path')  16551 images
     - images/train2012
     - images/train2007
     - images/val2012
     - images/val2007
    val: # val images (relative to 'path')  4952 images
     - images/test2007
    test: # test images (optional)
     - images/test2007
     
    # Classes
    nc: 1  # number of classes
    names: ['person']
    
    @return [data.yaml]'s path
    '''
    
    if not os.path.isdir(TASK_ID):
        os.makedirs(TASK_ID)
        
    if "DATA_PATH" in os.environ:
        dataset_pathes = os.environ["DATA_PATH"].split(":")
    else:
        datasets_path = os.environ["DATASETS_PATH"]
        dataset_pathes = datasets_path.split(":")
    

    if "DATA_SPLIT_RATIO" in os.environ:
        data_split_ratio = os.environ["DATA_SPLIT_RATIO"]
    else:
        data_split_ratio = "7:2:1"
    train_ratio, val_ratio, test_ratio = calRatio(data_split_ratio)
    
    
    print("process data path:", dataset_pathes)
    print("data_split_ratio:", data_split_ratio)

    # prepare train/val/test folders
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
    
    return "%s/_temp_data.yaml" % TASK_ID


def main():
    print("workid:", TASK_ID)
    data_yaml_path = prepareData()
    
    params = {
        "data":data_yaml_path,
        "imgsz": 320,
        "weights": "yolov5s.pt",
        "project": "./out/train",
        "noautoanchor": True,
        "optimizer": "Adam",
        "epochs": 300,
    }
    
    if "IMAGE_SIZE" in os.environ:
        imgsize = int(os.environ["IMAGE_SIZE"])
        params["imgsz"] = imgsize
    
    if "PRETRAIN" in os.environ:
        pretrain = os.environ["PRETRAIN"]
        params["weights"] = pretrain
    
    pretrain = params["weights"]
    if not os.path.exists(pretrain):
        try:
            shutil.copy("/root/pretrain/" + pretrain, pretrain)
        except Exception as e:
            print("pretrain load failed: ", e, " use yolov5s.pt instead")
            pretrain = "/root/pretrain/yolov5s.pt"
            pass
    
    if "MAX_EPOCHS" in os.environ:
        epochs = int(os.environ["MAX_EPOCHS"])
        params["epochs"] = epochs
    
    if "OPTIMIZER" in os.environ:
        optimizer = os.environ["OPTIMIZER"]
        params["optimizer"] = optimizer
    
    if "NO_AUTOANCHOR" in os.environ:
        no = os.environ["NO_AUTOANCHOR"]
        params["noautoanchor"] = no
    print(params)

    import train
    train.run(**params)
    pass

if __name__ == "__main__":
    main()