# SSD的极简实现

这个仓库存放了[SSD](http://arxiv.org/abs/1512.02325)的极简单实现，其只使用了pytorch和numpy来进行实现，其主要目的是为了通过复现一篇简单但是性能还不错的模型来入门目标检测，里面包含了目标检测中一些常用的概念，比如说anchor的生成以及匹配，iou,nms的实现等等，并且通过实现这些模块并拼接到一起实现了SSD算法。在实现之中唯一缺失的部分是数据增强部分，在训练的时候我仅仅使用了随机翻转一个策略，因为我的目的是为了学习目标检测中的一些知识，而不是为了提点，所以首先只要将模型能跑起来并且有一定的效果，就OK了，因为代码极少，因此对于初学者应该比较好上手。

# 代码组织
- train.py    实现了模型的训练部分的代码
- voc_dataset.py    voc数据的读取
- eval.py     在voc上计算MAP
- lib    包括ssd实现的一些模块
    * augmentations.py    数据增强
    * model.py    定义了网络模型
    * ssd_loss.py    定义了损失以及难样本挖掘
    * multibox_endoder.py    实现了anchor的生成以及怎么把这些anchor与一张图上的真实物体匹配上并编码
    * utils.py    一些工具，包括计算iou,nms等目标检测中常见的概念
    * voc_eval.py   测试voc分数的工具，不用管

- config.py    配置文件，所需要的所有参数都在这里面进行配置
- demo.ipynb    demo文件



# 准备工作
- 安装[Pytorch](https://pytorch.org),推荐使用[Anaconda](https://www.anaconda.com)环境，然后要安装pytorch的话(比如cuda9版本)只需要运行
```Shell
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
- 下载[VOC2007训练验证集](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)和[VOC2012训练验证集](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)以及[VOC2007测试集](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)，然后将VOC2007和VOC2012都放到同一个文件夹下面</br>
或者你是在linux系统下面并且能够联网的话,直接按顺序运行
```Shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCtest_06-Nov-2007.tar
```
你的路径下的结构将会是
```
~/VOCdevkit/
    -- VOC2007
    -- VOC2012
```
其中VOCdevkit就是你的VOC_ROOT

# 训练
- 首先从 https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth 下载VGG的预训练权重来作为我们基础网络的初始权重
```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
- 编辑config.py里面的文件，里面参数的含义都有注释，一般来说你需要关注的是VOC_ROOT，将其填写为准备工作里面你的VOC数据存放的地方
- 运行程序
```Shell
python train.py
```
或者在后台运行
```Shell
nohup python -u train.py &
watch -n 1 tail nohup.out

#按ctrl+c退出watch模式
```
因为在训练过程中开启了4个线程,因此如果想要杀死程序的话可以运行
```Shell
ps -ef|grep -w train|grep -v grep|cut -c 9-15|xargs kill -9
```

- 问题
    * 训练会使用GPU吗？ 如果你有GPU的话，程式会运行在cuda:0号卡上面,否则将会运行在cpu上面
    * OOM错误。  显存爆了，打开config.py减少batch size的大小
    * 训练出现了nan。   出现这个问题原因很多，试着改小学习率看看？打开config.py找到learning_rate改小


# Demo
我并没有在VOC测试集上面测试模型的准确度，但是仍然提供了一个demo来人肉查看检测的结果，可以使用你训练出来的模型，也可以使用</br>
链接:https://pan.baidu.com/s/1V2bRX1EogjU7JV_G2NRTDA  密码:u16q </br>
里面带有loss的模型来进行demo演示，其中loss越小代表训练的时间越久
```Shell
jupyter notebook
```
打开jupyter然后打开浏览器输入localhost:8888,打开demo.ipynb来查看
<img align="center" src= "https://github.com/HosinPrime/simple-ssd-for-beginners/blob/master/images/1.jpg" height = 400/> </br></br>
<img align="center" src= "https://github.com/HosinPrime/simple-ssd-for-beginners/blob/master/images/2.jpg" height = 400/>


# 测试
现在提供了测试代码在VOC2007 testset上面计算MAP,MAP的计算函数是从FaceBook开源的[Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py)里面拿过来的,然后eval部分的代码基本与demo之中保持一致.进行MAP的计算请运行
```Shell
python eval.py --model=weights/loss-1220.37.pth  --save_folder=result
```
程式运行完毕会在屏幕打印出各类别的map以及平均的map</br></br>
- 一些注意点</br>
其中model是你训练出的存放在weights里面用来测MAP的模型点,save_folder是存放每一类预测结果的文件夹,运行完程序后会生成一个result(--save_folder的参数)文件夹和一个annotations_cache文件夹，其中annotations_cache文件夹是MAP计算的时候存放真实标注临时信息的文件夹,由MAP计算函数产生
    * Todo
        * 尽可能训练一个map高的模型


# Results
|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
| [origin paper](http://arxiv.org/abs/1512.02325) |    0.772    |
|    this repo(eval using unofficial voc_eval code)    | 0.73-0.75 |


## 引用
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016](http://arxiv.org/abs/1512.02325).
- The code were mainly inspired by Those two repo, thanks for them for shareing us so elegant work
    * [pytorch.ssd](https://github.com/amdegroot/ssd.pytorch)
    * [Chainer](https://github.com/chainer/chainercv/tree/master/chainercv/links/model/ssd)

