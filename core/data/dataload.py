import os
import cv2
import csv
from PIL import Image
import torch
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import glob
import gc
import cv2
from torchvision import transforms

def prepare_dataset(data_path, ann_path, frame_len, prefix):
    datalist = []
    annlist = []
    ang = 0
    dis = 0
    fea = 0
    hap = 0
    sad = 0
    sur = 0
    neu = 0
    oth = 0
    ann_files = glob.glob(os.path.join(ann_path,"*.txt"))
    # transform = transforms.Compose([# transforms.ToPILImage(),
    #                                   transforms.Resize(size=(224, 224)),
    #                                 #   transforms.RandomHorizontalFlip(),
    #                                   # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #                                   # transforms.RandomErasing(),
    #                                   ])
    
    ann_files = ann_files[0:5]
    for file in tqdm(ann_files):
        print(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        frame_path = os.path.join(data_path, filename)
        img_num = 0
        data = []
        ann = []
        ann_new = []
        #1ファイルのデータとラベル取得
        with open(file, "r") as f:
            for line in f:
                if img_num != 0:
                    line = line.replace("\n","")
                    ann.append(line)
                else:
                    print(line)
                img_num = img_num + 1
        for i in range(len(ann)):
            img_file = os.path.join(frame_path, str(i).zfill(5)+".jpg")
            #print(img_file)
            if os.path.isfile(img_file):
                frame_bgr = cv2.imread(img_file)
                # frame = cv2.cvtColor(cv2.resize(frame_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(cv2.resize(frame_bgr, (112, 112)), cv2.COLOR_BGR2RGB)
                # img = Image.open(img_file)
                # frame = transform(frame_bgr)
                # print(frame.shape)
                if int(ann[i]) !=  -1:
                    data.append(frame)
                    # ann_new.append(torch.Tensor(int(ann[i])))
                    ann_new.append(int(ann[i]))
        # print(ann_new)
        if len(data) != len(ann_new):
            print("false_1")
            return 0
        print(len(data))
        print(len(data)//frame_len)
        for i in range(len(data)//frame_len):
            datalist.append(data[frame_len*i:frame_len*(i+1)])
            labels = ann_new[frame_len*i:frame_len*(i+1)]
            annlist.append(labels)
            # for i in labels:
            #     # print(i)
            #     if i == 0:
            #         neu += 1
            #     elif i == 1:
            #         ang += 1
            #     elif i == 2:
            #         dis += 1
            #     elif i == 3:
            #         fea += 1
            #     elif i == 4:
            #         hap += 1
            #     elif i == 5:
            #         sad += 1
            #     elif i == 6:
            #         sur += 1
            #     elif i == 7:
            #         oth += 1
            
        print(str(file)+" : finish")
        if len(datalist) != len(annlist):
            print("false_2")
            return 0
    del data
    del ann
    del ann_new
    del labels
    gc.collect()
    weight = np.array([neu,ang,dis,fea,hap,sad,sur,oth])
    # print(weight)
    weight = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    # max = np.max(weight)
    # weight = np.array([max/neu, max/ang, max/dis, max/fea, max/hap, max/sad, max/sur, max/oth])
    #[format(max/ang, '.2f'),format(max/neu, '.2f'),format(max/sad, '.2f'),format(max/hap, '.2f')]
    
    # print(np.array(datalist).shape)
    # print(np.array(annlist).shape)
    
    return np.array(datalist), np.array(annlist), weight