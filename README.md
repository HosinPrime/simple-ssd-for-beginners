# SSD : A very very simple ssd implementation using only pytorch and numpy

This repo contains some simple codes for me to learn the basic of object detection [中文请点击](https://github.com/HosinPrime/simple-ssd-for-beginners/blob/master/chinese.md), [SSD](http://arxiv.org/abs/1512.02325)(Single Shot MultiBox Detector) is a somewhat simple but powerful model to get started. So I try to implement it by myself, hoping I can get more insight in object dectection land. It's really amazing with deep learning and little code that machines can catch object show in the world. I try to reimplement it more readable and with clear codes. I hope this repo will help people who want to learn object detection and feel hard to get started.



# Code structure
- train.py    
- voc_dataset.py    
- eval.py
- lib    
    * augmentations.py    
    * model.py    
    * ssd_loss.py    
    * multibox_endoder.py    
    * utils.py    
    * voc_eval.py   

- config.py    
- demo.ipynb   


# getting started
- Install [Pytorch](https://pytorch.org), I recommand [Anaconda](https://www.anaconda.com) as your packge manager, and you can simplely install Pytorch by 
```Shell 
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch 
``` 
for example.

- download [VOC2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [VOC2012 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), download [VOC2007 testset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar), extract them and put them in a folder</br>
if you are using a linux machine, simple run
```Shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCtest_06-Nov-2007.tar
```
the structures would like
```
~/VOCdevkit/
    -- VOC2007
    -- VOC2012
```
then ~/VOCdevkit is your VOC root.


# Train
- for training ssd you need pretrained VGG weights as your basenet's starting point. so download this weight from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth, then put it in *weights* folder.
```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- vim config.py to change learning rate and batch size num...... and so on. It's not really neccesery, some thing you need to care about is VOC_ROOT, change it to your VOC root where you put your VOC data.
- A simple command is all you need
```Shell
python train.py
```
or
```Shell
nohup python -u train.py &
watch -n 1 tail nohup.out


#ctrl+c to quit!
```


- Question:
    * I have a GPU device, how do I use it? The code will detect that and use cuda:0 as default otherwise it use cpu
    * I get oom error. just vim config.py and reduce batch size
    * I get nan loss value.  your learning rate might be too large, try to set a lower learing rate
# Demo
I have not tested it on VOC dataset for I just reimplemented it for learning purpose, but there still provide a jupyter notebook for you to see the result,download the pretrained weights from https://drive.google.com/drive/folders/1XN-CXifL-2xilx9y8sb3Qmog_sbzW0k-?usp=sharing or use your own weights
```Shell
jupyter notebook
```
then go to localhost:8888 by default to see the demo.

<img align="center" src= "https://github.com/HosinPrime/simple-ssd-for-beginners/blob/master/images/1.jpg" height = 400/> </br></br>
<img align="center" src= "https://github.com/HosinPrime/simple-ssd-for-beginners/blob/master/images/2.jpg" height = 400/>



# Eval on VOC2007
Now I provide code to eval on VOC2007 testset, I use [Detectron's](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py) voc_eval.py to calculate MAP, to eval your model, just run
```Shell
python eval.py --model=weights/loss-1220.37.pth --save_folder=result
```
MAP result will show in your screen</br></br>
- something to notice</br>
--model is your model checkpoint to eval, after running those script a annotations_cache folder and a result(--save_folder) folder will show in this workspace. result folder contains prediction for each class.

# Results
|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
| [origin paper](http://arxiv.org/abs/1512.02325) |    0.772    |
|    this repo(eval using unofficial voc_eval code)    | 0.73-0.75 |


## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016](http://arxiv.org/abs/1512.02325).
- The code were mainly inspired by Those two repo, thanks for them for shareing us so elegant work
    * [pytorch.ssd](https://github.com/amdegroot/ssd.pytorch)
    * [Chainer](https://github.com/chainer/chainercv/tree/master/chainercv/links/model/ssd)
