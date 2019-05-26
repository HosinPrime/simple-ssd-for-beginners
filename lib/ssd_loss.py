

import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_negtives(logits, labels, pos, neg_radio):
    '''
    logits (batch, n, 21)
    labels (batch, n, )
    pos (batch, n,)
    
    挑选出在这一个batch中损失最大的负样本,因为在ssd中负样本的数量非常多,如果我们随机挑选负样本来进行训练的话,模型将会很快收敛,判别能力会变差,所以我们要选择那些最难辨别的负样本来训练
    '''
    
    num_batch, num_anchors, num_classes = logits.shape
    logits = logits.view(-1, num_classes)
    labels = labels.view(-1)
    
    #得到该anchor对应类别的损失
    losses = F.cross_entropy(logits, labels, reduction='none')

    losses = losses.view(num_batch, num_anchors)

    #过滤掉正样本,因为我们只挖掘负样本
    losses[pos] = 0

    #loss (batch, n)
    #下面通过两个argsort可以使我们得到该某一损失在排序后应该存在的位置
    #举例说明原来损失为(3,2,1,4)
    #通过下面的操作可以得到(1,2,3,0)
    #为什么呢？因为损失排序后为(4,3,2,1)
    #而3排序后在1号位置,2排序后再2号位置,1排序后在3号位置,4排序后再0号位置
    loss_idx = losses.argsort(1, descending=True)
    rank = loss_idx.argsort(1) #(batch, n)

    #通过得到batch每一个图片正样本的数量来根据比例为其选择负样本数量，最大不能超过anchors的数量
    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(neg_radio*num_pos, max=pos.shape[1]-1) #(batch, 1)
    neg = rank < num_neg.expand_as(rank)
    
    
    
    return neg
    
class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes=10, neg_radio=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_radio = neg_radio
    
    def forward(self, pred_loc, pred_label, gt_loc, gt_label):
        '''
        pred_loc : (batch, anchor_num, 4)
        pred_label : (batch, anchor_num, num_classes)
        gt_loc : (batch, anchor_num, 4)
        gt_label : (batch, anchor_num)
        '''

        num_batch = pred_loc.shape[0]

        #挑选正样本来进行坐标回归
        pos_idx = gt_label > 0
        pos_loc_idx = pos_idx.unsqueeze(2).expand_as(pred_loc)
        pred_loc_pos = pred_loc[pos_loc_idx].view(-1, 4)
        gt_loc_pos = gt_loc[pos_loc_idx].view(-1, 4)

        loc_loss = F.smooth_l1_loss(pred_loc_pos, gt_loc_pos, reduction='sum')

        
        #进行困难负样本挖掘
        logits = pred_label.detach()
        labels = gt_label.detach()
        neg_idx = hard_negtives(logits, labels, pos_idx, self.neg_radio) #neg (batch, n)

        #这里我们用于训练分类的是困难负样本和正样本 
        pos_cls_mask = pos_idx.unsqueeze(2).expand_as(pred_label)
        neg_cls_mask = neg_idx.unsqueeze(2).expand_as(pred_label)

        conf_p = pred_label[(pos_cls_mask+neg_cls_mask).gt(0)].view(-1, self.num_classes)
        target = gt_label[(pos_idx+neg_idx).gt(0)]

        cls_loss = F.cross_entropy(conf_p, target, reduction='sum')
        N = pos_idx.long().sum()

        loc_loss /= N
        cls_loss /= N


        return loc_loss, cls_loss
