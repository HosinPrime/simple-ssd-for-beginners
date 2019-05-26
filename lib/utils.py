

import numpy as np
import torch
import cv2




def point_form(boxes):
    '''
    将[cx, cy, w, h]类型转换成[xmin, ymin, xmax, ymax]形式
    '''

    tl = boxes[:, :2] - boxes[:, 2:]/2
    br = boxes[:, :2] + boxes[:, 2:]/2

    return np.concatenate([tl, br], axis=1)


def detection_collate(batch):
    '''
    因为每个样本的gt个数不一定,所以我们要自己定义拼接函数
    '''
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs), np.array(targets)



def bbox_iou(box_a, box_b):
    '''
    计算两个box数组的iou
    box_a : (m, 4)
    box_b : (n, 4)
    '''
    m = box_a.shape[0]
    n = box_b.shape[0]

    #broadcasting,这么做相当于(m,1,2)和(1,n,2)的进行操作,最后得到(m,n,2)大小
    tl = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    br = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])

    wh = np.maximum(br-tl, 0)
    
    inner = wh[:, :, 0]*wh[:, :, 1]

    a = box_a[:, 2:] - box_a[:, :2]
    b = box_b[:, 2:] - box_b[:, :2]

    a = a[:, 0] * a[:, 1]
    b = b[:, 0] * b[:, 1]

    a = a[:, None]
    b = b[None, :]

    #最后相当于(m,n) / (m, 1) + (1,n) - (m,n)
    #得到一个(m,n)的矩阵，里面每一个点(i,j)代表i和j的iou

    return inner / (a+b-inner)


def nms(boxes, score, threshold=0.4):
    '''
    通过iou来进行nms操作
    boxes : (n, 4)
    score: (n, )
    '''

    sort_ids = np.argsort(score)
    pick = []
    while len(sort_ids) > 0:
        #先拿一个概率最大的盒子
        i = sort_ids[-1]
        pick.append(i)
        if len(sort_ids) == 1:
            break

        #用这个盒子去与剩下的盒子进行iou计算，并删掉iou大于某一阈值的盒子,当剩下的盒子少于1个的时候结束
        sort_ids = sort_ids[:-1]
        box = boxes[i].reshape(1, 4)
        ious = bbox_iou(box, boxes[sort_ids]).reshape(-1)

        sort_ids = np.delete(sort_ids, np.where(ious > threshold)[0])

    return pick


def detect(locations, scores, w, h, nms_threshold=0.5, obj_threshold=0.1):
    '''
    
    This function will detect object from predicted locations accoding to their
    scores.
    
    locations (num_anchors, 4)
    scores (num_anchors, num_classes) 0 is background
    w, h is the src image's width and heights, ssd use it to map predlocations back
    to the src image
    
    return
    gt_labels (m,)
    gt_conf (m, )
    gt_location (m, 4)
    '''
    #这里得到的预测是(num_anchors, num_classes)大小的形式,为了找出每一个类别的物体，我们先进行转置操作
    #然后0号是背景，将其去除
    scores = scores.transpose(1, 0)[1:, :]
    num_classes = scores.shape[0]
    
    gt_conf = []
    gt_locations = []
    gt_labels = []
    for i in range(num_classes):
        #对于每一类物体先通过置信度排序然后去找置信度大于阈值的那些物体
        prob = scores[i]
        sort_idx = prob.argsort()[::-1]
        sort_prob = prob[sort_idx]
        j = 0
        one_score = []
        one_boxes = []
        one_labels = []
        while sort_prob[j] > obj_threshold:
            one_score.append(sort_prob[j])
            one_boxes.append(locations[sort_idx[j]])
            j += 1
        
        #nms
        if len(one_score) > 0:
            #将置信度较高的这些物体中iou较大的剔除
            one_score = np.array(one_score)
            one_boxes = np.array(one_boxes)

            pick = nms(one_boxes, one_score, nms_threshold)
            one_labels += [i]*len(pick)
            one_score = one_score[pick]
            #将归一化的预测通过原图大小返回去
            one_boxes = one_boxes[pick] * np.array([w, h, w, h])
            one_boxes = one_boxes.astype(int)
            
            gt_conf.extend(one_score)
            gt_locations.extend(one_boxes)
            gt_labels.extend(one_labels)
        
    return np.array(gt_labels), np.array(gt_conf), np.array(gt_locations)


def draw_rectangle(src_img, labels, conf, locations, label_map):
    '''
    src_img : 要画框的图片
    labels : 物体得标签,数字形式
    conf : 该处有物体的概率
    locations : 坐标
    label_map : 将labels映射回类别名字
    
    return
        画上框的图片
    '''
    num_obj = len(labels)
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    img = src_img.copy()
    for i in range(num_obj):
        tl = tuple(locations[i][:2])
        br = tuple(locations[i][2:])
        
        cv2.rectangle(img,
                      tl,
                      br,
                      COLORS[i%3], 3)
        cv2.putText(img, label_map[labels[i]], tl,
                    FONT, 1, (255, 255, 255), 2)
    
    #img = img[:, :, ::-1]

    return img
