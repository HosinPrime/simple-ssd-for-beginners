# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init



class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)
        

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x


def VGG():
    base_vgg = models.vgg16().features
    base_vgg[16].ceil_mode = True
    vgg = []
    for i in range(30):
        vgg.append(base_vgg[i])

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    vgg += [pool5, conv6,nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return vgg


def Extra():
    layers = []
    conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
    conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
    conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
    conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
    conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
    conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
    conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)

    layers = [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2, conv11_1, conv11_2]

    return layers


def Feature_extractor(vgg, extral, bboxes, num_classes):
    
    #坐标和分类
    loc_layers = []
    conf_layers = []
    
    #vgg的提取层
    vgg_useful = [21, 33]
    
    for k, v in enumerate(vgg_useful):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 bboxes[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        bboxes[k] * num_classes, kernel_size=3, padding=1)]
    
    #然后此时我们已经用掉了两个盒子，所以k从2开始,然后extra的提取层是每个2个一个所以简单的用enumerate(extra[1::2], 2)
    #表示k从2开始,而extra从1开始，步伐为2
    for k, v in enumerate(extral[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                  * num_classes, kernel_size=3, padding=1)]
        
    
    
    return loc_layers, conf_layers 




class SSD(nn.Module):

    def __init__(self, num_classes, bboxes):
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.bboxes = bboxes      #每个特征提取处的盒子数量

        self.vgg_list = VGG()
        self.extra_list = Extra()

        self.loc_layers_list, self.conf_layers_list = Feature_extractor(self.vgg_list, self.extra_list, self.bboxes, self.num_classes)

        self.L2Norm = L2Norm(512, 20)


        #将list转化为ModuleList使得其被包装成Module对象
        self.vgg = nn.ModuleList(self.vgg_list)
        self.extras = nn.ModuleList(self.extra_list)
        self.loc = nn.ModuleList(self.loc_layers_list)
        self.conf = nn.ModuleList(self.conf_layers_list)



    #list里面的东西毫无关联，我们要在前向传播中建立它们之间的关系

    def forward(self, x):

        #首先我们要在前向传播的过程中将特征提取层的输出保存下来，然后才能通过loc_layer和conf_layer去作用它们
        #同时我们想将输出的位置坐标和类别概率存储起来
        source = []
        loc = []
        conf = []


        #首先提取vgg的特征
        vgg_source = [22, 34]
        for i, v in enumerate(self.vgg):
            x = v(x)

            if i in vgg_source:
                if i == 22:
                    s = self.L2Norm(x)
                else:
                    s = x
                source.append(s)

        #提取extra的特征
        for i, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if i % 2 == 1:
                source.append(x)


        #对于每个特征图提取特征
        for s, l, c in zip(source, self.loc, self.conf):
            #输出是c,w,h的形式,转换成w,h,c的形式,pytorch中转置后用contiguous来使得内存连续
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())

        #因为这里的loc和conf是list所以我们将他转换为tensor
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        #然后将位置和类别展平，这样得到的数据大小是
        #loc  -- (batch, num_box*w*h, 4)
        #conf -- (batch, num_box*w*h, num_classes)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        return loc, conf



if __name__ == '__main__':

    x = torch.randn(1, 3, 300, 300) 
    ssd = SSD(21, [4,6,6,6,4,4])

    y = ssd(x)
    print(y[0].shape, y[1].shape)
