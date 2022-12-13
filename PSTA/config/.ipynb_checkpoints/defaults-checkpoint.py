from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = "0,1"
_C.MODEL.ARCH = "ResNet50"
_C.MODEL.NAME = 'resnet50'

_C.MODEL.SYN_BN = True

_C.MODEL.LAST_STRIDE = 1
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

#2019CVPR Bag of Tricks and Strong Baseline for Deep Person Re-identification
_C.MODEL.IF_WITH_CENTER = 'no' #center loss
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
#For example , if loss type is cross entropy loss + triplet loss + center loss
#the setting should be _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'
_C.MODEL.IF_LABELSMOOTH = 'on'
# _C.MODEL.LOCAL_RANK = [0]

#------------------------------------------------------------------------------------
# INPUT
#------------------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [384,128]
_C.INPUT.SIZE_TEST = [384,128]
_C.INPUT.PROB = 0.5  #Random probability for image horizontal filp
_C.INPUT.RE_PROB = 0.5  #Random probability for random erasing
_C.INPUT.PIXEL_MEAN = [0.485,0.456,0.406]
_C.INPUT.PIXEL_STD = [0.229,0.224,0.225]
_C.INPUT.PADDING = 10 #Value of padding size

#----------------------------------------
# Dataset
#----------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAME = 'mars'
_C.DATASETS.ROOT_DIR = '/home/wyq/dataset/video/mars'
_C.DATASETS.SEQ_LEN = 4
_C.DATASETS.TRAIN_SAMPLE_METHOD = "random"
_C.DATASETS.TEST_SAMPLE_METHOD = "dense"
_C.DATASETS.TEST_MAX_SEQ_NUM = 200
_C.DATASETS.ATTRIBUTE_LOSS = "bce"

#--------------------------------------------
#DataLoader
#--------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.SAMPLER = 'softmax'
_C.DATALOADER.NUM_INSTANCE = 2

#----------------------------------------------
#Solver
#----------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"

#Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
#Base learning rate
_C.SOLVER.BASE_LR = 3e-4
#Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3 #Margin of triplet loss
_C.SOLVER.CLUSTER_MARGIN = 0.3 #Margin of cluster

_C.SOLVER.CENTER_ON = 0
_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1

#Setting of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

#decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
#decay step of learning rate
_C.SOLVER.STEPS = (30,55)

#warm up factor
_C.SOLVER.WARMUP_FACTOR = 1.0/3
#iterations of warm up
_C.SOLVER.WARMUP_ITERS = 500
#method of warm up , option : 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

#epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

_C.SOLVER.FP_16 = True
_C.SOLVER.SEQS_PER_BATCH = 12

_C.TEST = CN()
#Number of images per batch during test
_C.TEST.SEQS_PER_BATCH = 128
#If test with re-rankoing , options :'yes','no'
_C.TEST.RE_RANKING = 'no'
#Path to trained model
_C.TEST.WEIGHT = ""
#Whether feature is nomalized before test , if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

#Method for temporal pooling in
_C.TEST.TEMPORAL_POOL_METHOD = 'avg'

#------------------------
#misc options
#------------------------
#Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
_C.RANDOM_SEED = 999
_C.EVALUATE_ONLY = False