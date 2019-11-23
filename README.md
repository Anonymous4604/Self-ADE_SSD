# Self-ADE 

Code for CVPR submission 4604

## Setup

Clone this repository:

```
git clone https://github.com/Anonymous4604/Self-ADE_SSD
cd Self-ADE_SSD
```

Download and compile apex:
Download:
```
git clone https://github.com/NVIDIA/apex.git
```

Compile:
```
cd apex 
python setup.py install --cuda_ext --cpp_ext
```

Install dependencies:
```
pip install -r requirements.txt
```

## Download data and pretrained model

### Datasets

Download VOC2007 and VOC2012 from: http://host.robots.ox.ac.uk/pascal/VOC/

Download clipart1k, comic2k and watercolor2k from: https://naoto0804.github.io/cross_domain_detection/

Prepare datasets in a directory called datasets. Structure:

```
Self-ADE_SSD:
 - datasets:
   - VOC2007:
     - JPEGImages
     - ImageSets
     - Annotations
   - VOC2012:
     - ...
   - clipart:
     - ...
   - comic:
     - ...
   - watercolor:
     - ...
```

### Pretrained Model

We provide one model pre-trained on VOC2007 trainval + VOC2012 trainval (detection and rotation tasks).

```
cat checkpoints/pretrained* > checkpoints/pretrained.tar.gz
tar -C checkpoints -zxvf checkpoints/pretrained.tar.gz
```

## Run

Test the pre-trained model on clipart to check mAP

```
python eval_ssd.py --config-file configs/clipart_self_ade.yaml --weights checkpoints/pretrained.pth
```

Run Adaptive evaluation on clipart using the pre-trained model

```
python self_ade.py --config-file configs/clipart_self_ade.yaml --weights checkpoints/pretrained.pth
--warmup_step 20 --self_ade_iterations 50 MODEL.SELF_SUPERVISOR.SELF_ADE_BREAKPOINTS 21,25,30,35,40,
```

The code will output mAP values for different numbers of iterations.
To experiment on a different target, switch clipart with comic or watercolor.
