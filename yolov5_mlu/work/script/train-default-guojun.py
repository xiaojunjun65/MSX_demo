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

def get_sub_file_list(_dir):
    files = []
    if os.path.isfile(_dir):
        files.append(_dir)
    else:
        for dirpath, _, filenames in os.walk(_dir):
            for filename in filenames:
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.jpeg'):
                    files.append(os.path.join(dirpath, filename))
    return files


def get_data_list(_dir):
    _train_file_list = []
    _val_file_list = []
    _test_file_list = []
    _other_file_list = []
    if os.path.isfile(_dir):
        _other_file_list.append(_dir)
    elif os.path.isdir(_dir):
        for filename in os.listdir(_dir):
            fullpath = os.path.join(_dir, filename)
            if os.path.isfile(fullpath):
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.jpeg'):
                    _other_file_list.append(fullpath)
            else:
                if filename == 'train':
                    _train_file_list.extend(get_sub_file_list(fullpath))
                elif filename == 'val':
                    _val_file_list.extend(get_sub_file_list(fullpath))
                elif filename == 'test':
                    _test_file_list.extend(get_sub_file_list(fullpath))
                else:
                    _train_file_list_sub, _val_file_list_sub, _test_file_list_sub, _other_file_list_sub = get_data_list(fullpath)
                    _train_file_list.extend(_train_file_list_sub)
                    _val_file_list.extend(_val_file_list_sub)
                    _test_file_list.extend(_test_file_list_sub)
                    _other_file_list.extend(_other_file_list_sub)
    return _train_file_list, _val_file_list, _test_file_list, _other_file_list


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
    # train_path=[]
    # val_path=[]
    # test_path=[]

    # plain_files = []
    # for dpath in dataset_pathes:
    #     if os.path.isdir(dpath):
    #         for spath in os.listdir(dpath):
    #             fullpath = os.path.join(dpath, spath)
    #             if spath == "train":
    #                 train_path.append(fullpath)
    #             elif spath == "val":
    #                 val_path.append(fullpath)
    #             elif spath == "test":
    #                 test_path.append(fullpath)
    #             elif spath.endswith(".jpg") or spath.endswith(".png"):
    #                 plain_files.append(fullpath)
    #     elif os.path.isfile(dpath):
    #         pass
    
    # if len(plain_files) > 10:
    #     random.shuffle(plain_files)
        
    #     train_split_idx = len(plain_files)*train_ratio
    #     val_split_idx = len(plain_files)*(train_ratio+val_ratio)
        
    #     trainf = open("%s/_temp_train.txt"%(TASK_ID), "w")
    #     valf = open("%s/_temp_val.txt"%(TASK_ID), "w")
    #     testf = open("%s/_temp_test.txt"%(TASK_ID), "w")
    #     for i, pp in enumerate(plain_files):
    #         if i < train_split_idx:
    #             trainf.write("%s\n" % (pp))
    #         elif i < val_split_idx:
    #             valf.write("%s\n" % (pp))
    #         else:
    #             testf.write("%s\n" % (pp))
                
    #     trainf.close()
    #     valf.close()
    #     testf.close()
        
    #     train_path.append("%s/_temp_train.txt"%(TASK_ID))
    #     val_path.append("%s/_temp_val.txt"%(TASK_ID))
    #     test_path.append("%s/_temp_test.txt"%(TASK_ID))
    train_file_list = []
    val_file_list = []
    test_file_list = []
    other_file_list = []

    # 遍历所有文件夹中的图片文件
    for dpath in dataset_pathes:
        _train_file_list_sub, _val_file_list_sub, _test_file_list_sub, _other_file_list_sub = get_data_list(dpath)
        train_file_list.extend(_train_file_list_sub)
        val_file_list.extend(_val_file_list_sub)
        test_file_list.extend(_test_file_list_sub)
        other_file_list.extend(_other_file_list_sub)
    # train或者val只有一个, 所有数据全部重新分配
    if len(train_file_list) == 0 or len(val_file_list) == 0:
        other_file_list.extend(train_file_list)
        other_file_list.extend(val_file_list)
        other_file_list.extend(test_file_list)
        train_file_list = []
        val_file_list = []
        test_file_list = []

    if len(other_file_list) > 0:
        # 带分配数据打乱
        random.shuffle(other_file_list)
        total_len = len(train_file_list) + len(val_file_list) + len(test_file_list) + len(other_file_list)
        print('train_file_list:', len(train_file_list))
        print('val_file_list:', len(val_file_list))
        print('test_file_list:', len(test_file_list))
        print('other_file_list:', len(other_file_list))
        print('total_len:', total_len)
        train_split_idx = 0
        val_split_idx = 0
        test_split_idx = 0
        other_file_list_len = len(other_file_list)
        # train数量不足
        assigned_train_len = int(total_len * train_ratio)
        if len(train_file_list) < assigned_train_len:
            train_split_idx = assigned_train_len - len(train_file_list)
            if train_split_idx > other_file_list_len:
                train_split_idx = other_file_list_len
            train_file_list.extend(other_file_list[:train_split_idx])
        # val数量不足
        assigned_val_len = int(total_len * val_ratio)
        if len(val_file_list) < assigned_val_len:
            val_split_idx = assigned_val_len - len(val_file_list) + train_split_idx
            if val_split_idx > other_file_list_len:
                val_split_idx = other_file_list_len
            val_file_list.extend(other_file_list[train_split_idx: val_split_idx])
        # test数量不足
        assigned_test_len = int(total_len * test_ratio)
        if len(test_file_list) < assigned_test_len:
            test_split_idx = assigned_test_len - len(test_file_list) + val_split_idx
            if test_split_idx > other_file_list_len:
                test_split_idx = other_file_list_len
            test_file_list.extend(other_file_list[val_split_idx: test_split_idx])
        # 上述分完了，还有剩余
        if test_split_idx < other_file_list_len:
            left_len = other_file_list_len - test_split_idx
            left_train_split_idx = int(left_len * train_ratio)
            val_split_idx = int(left_len * (train_ratio + val_ratio))
            train_file_list.extend(other_file_list[test_split_idx: test_split_idx + left_train_split_idx])
            val_file_list.extend(other_file_list[test_split_idx + left_train_split_idx: test_split_idx + val_split_idx])
            test_file_list.extend(other_file_list[test_split_idx + val_split_idx:])
    print('final train_file_list:', len(train_file_list))
    print('final val_file_list:', len(val_file_list))
    print('final test_file_list:', len(test_file_list))
    print('final total_len:', len(train_file_list) + len(val_file_list) + len(test_file_list))
    # 文件列表写入文件
    with open("%s/train.txt" % (TASK_ID), "w") as f:
        for item in train_file_list:
            f.write("%s\n" % (item))
    with open("%s/val.txt" % (TASK_ID), "w") as f:
        for item in val_file_list:
            f.write("%s\n" % (item))
    with open("%s/test.txt" % (TASK_ID), "w") as f:
        for item in test_file_list:
            f.write("%s\n" % (item))

    train_path = [("%s/train.txt " % (TASK_ID))]
    val_path = [("%s/val.txt " % (TASK_ID))]
    test_path = [("%s/test.txt " % (TASK_ID))]
    
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
    
    return "%s/_temp_data.yaml" % TASK_ID,labels


def main():
    print("workid:", TASK_ID)
    data_yaml_path,labels = prepareData()
    
    params = {
        "data":data_yaml_path,
        "imgsz": 320,
        "project": "./out/train",
        "noautoanchor": True,
        "optimizer": "Adam",
        "epochs": 300,
        "label_name":labels,
        "device":"mlu"
    }
    
    if "IMAGE_SIZE" in os.environ:
        imgsize = int(os.environ["IMAGE_SIZE"])
        params["imgsz"] = imgsize
    
    # if "PRETRAIN" in os.environ:
    #     pretrain = os.environ["PRETRAIN"]
    #     params["weights"] = pretrain
    
    # pretrain = params["weights"]
    # if not os.path.exists(pretrain):
    #     try:
    #         shutil.copy("/root/pretrain/" + pretrain, pretrain)
    #     except Exception as e:
    #         print("pretrain load failed: ", e, " use yolov5s.pt instead")
    #         pretrain = "/root/pretrain/yolov5s.pt"
    #         pass
    
    if "MAX_EPOCHS" in os.environ:
        epochs = int(os.environ["MAX_EPOCHS"])
        params["epochs"] = epochs
    
    if "OPTIMIZER" in os.environ:
        optimizer = os.environ["OPTIMIZER"]
        params["optimizer"] = optimizer
    
    if "NO_AUTOANCHOR" in os.environ:
        no = os.environ["NO_AUTOANCHOR"]
        params["noautoanchor"] = no
    
    
    import train
    train.run(**params)
    pass

if __name__ == "__main__":
    main()