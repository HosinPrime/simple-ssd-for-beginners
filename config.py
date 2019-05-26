




class Config:

	
    #存放voc数据集的地址
    VOC_ROOT = '/unsullied/sharefs/liuhaoxin/isilon-home/data/VOC/VOCdevkit'

    #分类的类别数(包括了背景算一类)
    num_classes = 21

    #如果你想从已经训练的掉开始训练则取消注释下一行,从头训练的话保持原样
    #resume = 'weights/latest.pth'
    resume = None
    
    #下面是一些常用模型训练的参数,根据需要进行更改
    
    #学习率,学习率太大可能会导致损失nan
    lr = 0.001
    #batch size大小,太大可能导致爆显存
    batch_size = 32 
    
    #下面这两参数一般不用改,用于梯度加速和防止过拟合
    momentum = 0.9
    weight_decay = 5e-4

    #训练多少个epcoch
    epoch = 100 

    #这两个参数可以进行调节,lr_reduce_epoch是多少个epoch后学习率乘上gamma
    #比如gamma为0.1,lr_reduce_epoch=30即每隔30个epoch，学习率降低10倍
    gamma = 0.2
    lr_reduce_epoch = 30


    #模型保存的文件夹，在该文件家中主干网参数的名称
    save_folder = 'weights/'
    basenet = 'vgg16_reducedfc.pth'

    #每隔多少个batch答应一次信息
    log_fn = 10 


    #负样本比例
    neg_radio = 3
    

    #ssd中的一些参数,以下参数是基于论文给出的，谨慎修改！！
    
    #输入图片的大小
    min_size = 300
    #ssd中共有6个特征层,它们的大小依次如下，用于anchor的生成
    grids = (38, 19, 10, 5, 3, 1)
    #每个特征层anchor进行缩放的比例
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    #原图片大小(300)相较于特征层缩小了多少倍 比如(300 / 38 约等于8)
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    #每一个特征层上面防止的anchor的大小，原则上是特征图越小放置越大
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)] 
    #每个特征图上anchor的数量,用于模型建造时候最后一层的channel的产生
    anchor_num = [4, 6, 6, 6, 4, 4]

    #图片每个通道上面的均值
    mean = (104, 117, 123)
    #用于坐标编码的方差
    variance = (0.1, 0.2)



opt = Config()
