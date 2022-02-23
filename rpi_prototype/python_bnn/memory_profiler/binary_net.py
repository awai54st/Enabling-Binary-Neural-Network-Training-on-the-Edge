from core.numpy_models.full_precision_models.layers import Flatten, Softmax
from core.losses import CrossEntropy
from core.optimizers import GradientDescent, Adam
from core.label_utils import label_binerizer
from core.numpy_models.full_precision_models.pooling_layers import MaxPooling

from core.numpy_models.xnor_models.bit_packed_layers import XNorDense, XNorConv2D, BatchNorm
from core.numpy_models.binary_models.activation_layers import BinaryActivation
from core.model import Sequential
import numpy as np

def get_model_32():
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = 100
    LR_ADAM=np.float32(1e-3) # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    LR_SGD=1e-1 # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    DTYPE = np.float32
    
    model_layers = [
        # 128
        XNorConv2D(128, (3,3), padding="same", dtype=DTYPE, random_seed=0, first_layer=True),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        XNorConv2D(128, (3,3), padding="same", dtype=DTYPE, random_seed=1),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        # 256
        XNorConv2D(256, (3,3), padding="same", dtype=DTYPE, random_seed=2),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        XNorConv2D(256, (3,3), padding="same", dtype=DTYPE, random_seed=3),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        # 512
        XNorConv2D(512, (3,3), padding="same", dtype=DTYPE, random_seed=4),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        XNorConv2D(512, (3,3), padding="same", dtype=DTYPE, random_seed=5),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        Flatten(dtype=DTYPE), 
        XNorDense(1024, dtype=DTYPE, random_seed=0),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(1024, dtype=DTYPE, random_seed=1),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(10, dtype=DTYPE, random_seed=2),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        Softmax(dtype=DTYPE)
    ]


    loss_op = CrossEntropy(dtype=DTYPE)
    optimizer_op = Adam(lr=LR_ADAM)

    BinaryNet_numpy = Sequential(model_layers)
    BinaryNet_numpy.compile(loss_op, optimizer_op)
    return BinaryNet_numpy



def get_model_16():
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = 100
    LR_ADAM=np.float32(1e-3) # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    LR_SGD=1e-1 # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    DTYPE = np.float32
    
    model_layers = [
        # 128
        XNorConv2D(128, (3,3), padding="same", dtype=DTYPE, random_seed=0, first_layer=True),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        XNorConv2D(128, (3,3), padding="same", dtype=DTYPE, random_seed=1),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        # 256
        XNorConv2D(256, (3,3), padding="same", dtype=DTYPE, random_seed=2),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        XNorConv2D(256, (3,3), padding="same", dtype=DTYPE, random_seed=3),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        # 512
        XNorConv2D(512, (3,3), padding="same", dtype=DTYPE, random_seed=4),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        XNorConv2D(512, (3,3), padding="same", dtype=DTYPE, random_seed=5),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        Flatten(dtype=DTYPE), 
        XNorDense(1024, dtype=DTYPE, random_seed=0),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(1024, dtype=DTYPE, random_seed=1),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(10, dtype=DTYPE, random_seed=2),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        Softmax(dtype=DTYPE)
    ]


    loss_op = CrossEntropy(dtype=DTYPE)
    optimizer_op = Adam(lr=LR_ADAM)

    BinaryNet_numpy = Sequential(model_layers)
    BinaryNet_numpy.compile(loss_op, optimizer_op)
    return BinaryNet_numpy