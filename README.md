# Self-ADE 

Code for CVPR submission 4604

## Setup

Clone this repository:

```
git clone https://github.com/Anonymous4604/Self-ADE_SSD
```

Install pytorch
```
pip install torch==1.2.0
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
cd ../Self-ADE_SSD
pip install -r requirements.txt
```

## Download data

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

## Run

To pre-train your own model dowload ImageNet vgg from:

```
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

then run
```
python train_ssd.py --config-file configs/baseline_rs.yaml --vgg vgg16_reducedfc.pth
```

Test the pre-trained model on clipart to check mAP

```
python eval_ssd.py --config-file configs/clipart_self_ade.yaml --weights /path/to/last/checkpoint.pth
```

Run Adaptive evaluation on clipart using the pre-trained model

```
python self_ade.py --config-file configs/clipart_self_ade.yaml --weights /path/to/last/checkpoint.pth
--warmup_step 20 --self_ade_iterations 50 MODEL.SELF_SUPERVISOR.SELF_ADE_BREAKPOINTS 21,25,30,35,40,
```

The code will output mAP values for different numbers of iterations.
To experiment on a different target, switch clipart with comic or watercolor.
