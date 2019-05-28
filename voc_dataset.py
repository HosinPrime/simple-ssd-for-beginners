





import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np

import os
import cv2
from config import opt

from lib.augmentations import preproc_for_test, preproc_for_train

VOC_LABELS = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    )



class VOCDetection(data.Dataset):


    def __init__(self, opt, image_sets=[['2007', 'trainval'], ['2012', 'trainval']], is_train=True):


        #你的voc root
        self.root = opt.VOC_ROOT
        #使用的数据集列表,每个数据集包括年份和使用的部分
        self.image_sets = image_sets
        self.is_train = is_train
        self.opt = opt    #我们需要知道在预处理的时候要将图片resize到多大以及减去的方差等信息
    
        self.ids = []
        #遍历数据集将图片得路径加到id里面
        for (year, name) in self.image_sets:
            root_path = os.path.join(self.root, 'VOC' + year)
            ano_file = os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')
    
            with open(ano_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    ano_path = os.path.join(root_path, 'Annotations', line + '.xml')
                    img_path = os.path.join(root_path, 'JPEGImages', line + '.jpg')
                    self.ids.append((img_path, ano_path))



    
    
    def __getitem__(self, index):
        img_path, ano_path = self.ids[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        boxes, labels = self.get_annotations(ano_path)
        
        if self.is_train:
            image, boxes, labels = preproc_for_train(image, boxes, labels, opt.min_size, opt.mean)
            image = torch.from_numpy(image)
           
        
        
        target = np.concatenate([boxes, labels.reshape(-1,1)], axis=1)
        
        return image, target



    def get_annotations(self, path):
        
        tree = ET.parse(path)

        #得到真实坐标和标签
        boxes = []
        labels = []
        
        for child in tree.getroot():
            if child.tag != 'object':
                continue

            bndbox = child.find('bndbox')
            box =[
                float(bndbox.find(t).text) - 1
                for t in ['xmin', 'ymin', 'xmax', 'ymax']
            ]


            label = VOC_LABELS.index(child.find('name').text) 
            
            boxes.append(box)
            labels.append(label)


        return np.array(boxes), np.array(labels)
            

        

    def __len__(self):
        return len(self.ids)
        
