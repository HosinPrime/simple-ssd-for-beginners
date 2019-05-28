


from voc_dataset import VOCDetection
from config import opt
import numpy as np
from lib.model import SSD
import torch
import torch.nn.functional as F
import os 
from lib.multibox_encoder import MultiBoxEncoder
from lib.ssd_loss import MultiBoxLoss
import cv2
from lib.utils import nms
from lib.augmentations import preproc_for_test
import matplotlib.pyplot as plt
from lib.utils import detect
from voc_dataset import VOC_LABELS
import tqdm
import os
import argparse

from lib.voc_eval import voc_eval





parser = argparse.ArgumentParser()

parser.add_argument('--model', 
                    default='weights/VOC_LAST.pth',
                    type=str,
                    help='model checkpoint used to eval VOC dataset')

parser.add_argument('--save_folder',
                    default='result',
                    type=str,
                    help='eval result save folder')

args = parser.parse_args()

output_dir = args.save_folder
checkpoint = args.model





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

voc_root = opt.VOC_ROOT
annopath = os.path.join(voc_root, 'VOC2007', 'Annotations', "%s.xml")  #真实的标签坐标所在的地方 
imgpath = os.path.join(voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')    #图片所在的地方
imgsetpath = os.path.join(voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')   #测试集的txt文件存在的地方
cachedir = os.path.join( os.getcwd(), 'annotations_cache')    #暂存这些信息的地方






if __name__ == '__main__': 
    
    print('using {} to eval, use cpu may take an hour to complete !!'.format(device))
    model = SSD(opt.num_classes, opt.anchor_num)
    
    print('loading checkpoint from {}'.format(checkpoint))
    state_dict = torch.load(checkpoint, map_location=None if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    print('model loaded')
    
    multibox_encoder = MultiBoxEncoder(opt)
    
    image_sets = [['2007', 'test']]
    test_dataset = VOCDetection(opt, image_sets=image_sets, is_train=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = [
        open(
            os.path.join(
                output_dir, '{:s}.txt'.format(label)),
            mode='w')
        for label in VOC_LABELS]

    print('start detect.........')

    for i in tqdm.tqdm(range(len(test_dataset))):

        src = test_dataset[i][0]
        img_name = os.path.basename(test_dataset.ids[i][0]).split('.')[0]
        image = preproc_for_test(src, opt.min_size, opt.mean)
        image = torch.from_numpy(image).to(device)
        with torch.no_grad():
            loc, conf = model(image.unsqueeze(0))
        loc = loc[0]
        conf = conf[0]
        conf = F.softmax(conf, dim=1)
        conf = conf.cpu().numpy()
        loc = loc.cpu().numpy()

        decode_loc = multibox_encoder.decode(loc)
        gt_boxes, gt_confs, gt_labels = detect(decode_loc, conf, nms_threshold=0.45, gt_threshold=0.1)

        #no object detected
        if len(gt_boxes) == 0:
            continue

        h, w = src.shape[:2]
        gt_boxes[:, 0] = gt_boxes[:, 0] * w
        gt_boxes[:, 1] = gt_boxes[:, 1] * h
        gt_boxes[:, 2] = gt_boxes[:, 2] * w
        gt_boxes[:, 3] = gt_boxes[:, 3] * h


        for box, label, score in zip(gt_boxes, gt_labels, gt_confs):
            print(img_name, score, *box, file=files[label])


    for f in files:
        f.close()
    
    

    print('start cal MAP.........')
    aps = []
    for f in os.listdir(output_dir):
        filename = os.path.join(output_dir, f)
        class_name = f.split('.')[0]
        rec, prec, ap = voc_eval(filename, annopath, imgsetpath.format('test'), class_name, cachedir, ovthresh=0.5, use_07_metric=True)
        print(class_name, ap)
        aps.append(ap)

    print('mean MAP is : ', np.mean(aps))
    



