

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




def detect(locations, scores, nms_threshold, gt_threshold):
    '''
    locations : decode过后的坐标 (num_anchors, 4)
    scores : 预测出来的分数 (num_anchors, 21)
    nms_threshold : nms的阈值
    gt_threshold : 认为是真实物体ground truth的阈值
    '''

    scores = scores[:, 1:] #第0类是背景,过滤掉

    #存放最后保留物体的信息,他的坐标,置信度以及属于哪一类
    keep_boxes = []
    keep_confs = []
    keep_labels = []
    
    #对每一类进行检测
    for i in range(scores.shape[1]):
        #得到这一类中可能有物体的anchor
        mask = scores[:, i] >= gt_threshold
        label_scores = scores[mask, i] 
        label_boxes = locations[mask]
        #没有找到直接下一个类别
        if len(label_scores) == 0:
            continue

        #进行nms
        pick = nms(label_boxes, label_scores, threshold=nms_threshold)
        label_scores = label_scores[pick]
        label_boxes = label_boxes[pick]
        

        keep_boxes.append(label_boxes.reshape(-1))
        keep_confs.append(label_scores)
        keep_labels.extend([i]*len(label_scores))
    
    #没有找到任何一个物体
    if len(keep_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
        
    
    keep_boxes = np.concatenate(keep_boxes, axis=0).reshape(-1, 4)

    keep_confs = np.concatenate(keep_confs, axis=0)
    keep_labels = np.array(keep_labels).reshape(-1)
#     print(keep_boxes.shape)
#     print(keep_confs.shape)
#     print(keep_labels.shape)

    return keep_boxes, keep_confs, keep_labels





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
    
    img = img[:, :, ::-1]

    return img