

import itertools
import numpy as np

from lib.utils import bbox_iou
from lib.utils import point_form

class MultiBoxEncoder(object):

    '''
    通过feature map的大小生成对应的anchors,在ssd当中因为使用了多尺度的feature map,所以为了实现方便
    我们通常都将生成的anchor(cx, cy, w, h)通过它所在的feature map的大小归一化到(0,1)之间
    '''

    def __init__(self, opt):
        self.variance = opt.variance

        default_boxes = list()
        for k in range(len(opt.grids)):

            #生成网格的中心点
            for v, u in itertools.product(range(opt.grids[k]), repeat=2):
                cx = (u + 0.5) * opt.steps[k]
                cy = (v + 0.5) * opt.steps[k]

                s = opt.sizes[k]
                default_boxes.append((cx, cy, s, s))

                s = np.sqrt(opt.sizes[k] * opt.sizes[k + 1])
                default_boxes.append((cx, cy, s, s))

                s = opt.sizes[k]
                for ar in opt.aspect_ratios[k]:
                    default_boxes.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    default_boxes.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))

        default_boxes = np.clip(default_boxes, a_min=0, a_max=1)
        self.default_boxes = np.array(default_boxes)

    def encode(self, boxes, labels, threshold=0.5):
        '''
        接收图片中物体所在的位置和它们对应的标签，为每一个anchor分配一个物体
        '''
        if len(boxes) == 0:
            return (
                np.zeros(self.default_boxes.shape, dtype=np.float32),
                np.zeros(self.default_boxes.shape[:1], dtype=np.int32))

        #计算iou
        iou = bbox_iou(point_form(self.default_boxes), boxes)


        #anchor匹配与它iou最大的真实物体
        #其实有很多anchor与所有真实物体都是iou=0,但是无所谓我们先让它们匹着
        gt_idx = iou.argmax(axis=1)
        iou = iou.max(axis=1)
        boxes = boxes[gt_idx]
        labels = labels[gt_idx]

        #我们要进行回归的是anchor与真实物体的一些差距，所以先进行编码
        loc = np.hstack((
            ((boxes[:, :2] + boxes[:, 2:]) / 2 - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_boxes[:, 2:]),
            np.log((boxes[:, 2:] - boxes[:, :2]) / self.default_boxes[:, 2:]) /
            self.variance[1]))

        #这里我们将背景定义为0号标签，通过iou将那些anchor与真实物体iou小于threshold的anchor标签
        #设置为0，通过这一步我们拥有了这个标签后就能知道哪些anchor是正样本
        conf = 1 + labels
        conf[iou < threshold] = 0
       

        return loc.astype(np.float32), conf.astype(np.int32)

    def decode(self, loc):
        '''
        因为我们回归的是编码以后的参数,所以预测出来的坐标要通过anchors来解码,为
        上一步编码的反动作
        '''
        boxes = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes


if __name__ == '__main__':
    from config import opt
    mb = MultiBoxEncoder(opt)

    print(mb.default_boxes[:10])
    



