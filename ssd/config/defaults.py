from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
_C.MODEL.THRESHOLD = 0.5
_C.MODEL.NUM_CLASSES = 21
# Hard negative mining
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2
_C.MODEL.SELF_SUPERVISED = False
_C.MODEL.SELF_SUPERVISOR = CN()
_C.MODEL.SELF_SUPERVISOR.TYPE = "rotation" 
_C.MODEL.SELF_SUPERVISOR.CLASSES = 4
_C.MODEL.SELF_SUPERVISOR.WEIGHT = 0.8
_C.MODEL.SELF_SUPERVISOR.SELF_ADE_BREAKPOINTS = ()
_C.MODEL.SELF_SUPERVISOR.DROPOUT = 0.0

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
_C.MODEL.PRIORS = CN()
_C.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
_C.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
_C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
_C.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
_C.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
_C.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Image size
_C.INPUT.IMAGE_SIZE = 300
# Values to be used for image normalization, RGB layout
# _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_MEAN = [123, 117, 104]

_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
# domain generalization settings
_C.DATASETS.DG = False
_C.DATASETS.DG_SETTINGS = CN()
_C.DATASETS.DG_SETTINGS.DEFAULT_DOMAIN = ()
_C.DATASETS.DG_SETTINGS.SOURCE_DOMAINS = ()

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# train configs
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.45
_C.TEST.CONFIDENCE_THRESHOLD = 0.01
# change MAX_PER_CLASS to 400 as official caffe code will slightly increase mAP(0.8025=>0.8063, 0.7783=>0.7798)
_C.TEST.MAX_PER_CLASS = 200
_C.TEST.MAX_PER_IMAGE = -1
_C.TEST.MODE = "split"  # In case of "split" each test set is evaluated independently,  in case of "joint" they are evaluated together

_C.OUTPUT_DIR = 'output'

_C.USE_AMP = False
