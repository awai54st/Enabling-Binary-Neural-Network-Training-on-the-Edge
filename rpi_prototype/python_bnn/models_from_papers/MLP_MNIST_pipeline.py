import sys
import os
sys.path.insert(0, os.path.abspath("../"))

#import tensorflow.keras.datasets.mnist as mnist
from sklearn.datasets import fetch_openml
import numpy as np
import time
import argparse

from core.label_utils import label_binerizer
from core.losses import CrossEntropy
from core.optimizers import GradientDescent, Adam
from core.model import Sequential
from core.numpy_models.full_precision_models.layers import Flatten, Softmax

# Binary layers
from core.numpy_models.binary_models.layers import BinaryDense, BinaryConv2D, BatchNorm as BinaryBatchNorm
from core.numpy_models.binary_models.activation_layers import BinaryActivation

# XNor layers
from core.numpy_models.xnor_models.bit_packed_layers import XNorDense, XNorConv2D, BatchNorm


def vanilla_BNN(BATCH_NORM_MOMENTUM, LR_ADAM, DTYPE):
    model_layers =  [
        Flatten(), 
        BinaryDense(256, dtype=DTYPE, random_seed=0),
        BinaryBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(256, dtype=DTYPE, random_seed=1),
        BinaryBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(256, dtype=DTYPE, random_seed=2),
        BinaryBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(256, dtype=DTYPE, random_seed=3),
        BinaryBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(10, dtype=DTYPE, random_seed=4),
        BinaryBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        Softmax(dtype=DTYPE)
    ]
    loss_op = CrossEntropy(dtype=DTYPE)
    optimizer_op = Adam(lr=LR_ADAM, dtype=DTYPE)

    LeNet5_numpy = Sequential(model_layers)
    LeNet5_numpy.compile(loss_op, optimizer_op)
    return LeNet5_numpy


def proposed_BNN(BATCH_NORM_MOMENTUM, LR_ADAM, DTYPE):
    model_layers = [
        Flatten(), 
        XNorDense(256, dtype=np.float32, first_layer=True, random_seed=0),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(256, dtype=DTYPE, random_seed=1),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(256, dtype=DTYPE, random_seed=2),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(256, dtype=DTYPE, random_seed=3),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        XNorDense(10, dtype=DTYPE, random_seed=4),
        BatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        Softmax(dtype=DTYPE)
    ]
    loss_op = CrossEntropy(dtype=DTYPE)
    optimizer_op = Adam(lr=LR_ADAM, dtype=DTYPE)

    LeNet5_numpy = Sequential(model_layers)
    LeNet5_numpy.compile(loss_op, optimizer_op)
    return LeNet5_numpy

def main(use_proposed):
    if use_proposed:
        print("Train MLP using proposed BNN method")
    else:
        print("Train MLP using vanilla BNN method")
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = 100
    LR_ADAM=1e-3 # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    LR_SGD=1e-1 # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    DTYPE = np.float32

    X_train = np.load("./mnist_data/mnist_X_train.npy")/255
    y_train = label_binerizer(np.load("./mnist_data/mnist_y_train.npy"), n_class=10)
    X_test = np.load("./mnist_data/mnist_X_test.npy")/255
    y_test = label_binerizer(np.load("./mnist_data/mnist_y_test.npy"), n_class=10)

    print("Training samples: ", X_train.shape)
    print("Testing samples: ", X_test.shape)

    if use_proposed:
        LeNet5_numpy = proposed_BNN(BATCH_NORM_MOMENTUM, LR_ADAM, DTYPE=np.float16)
    else:
        LeNet5_numpy = vanilla_BNN(BATCH_NORM_MOMENTUM, LR_ADAM, DTYPE=np.float32)

    for epoch in range(100):
        loss_batch = np.array([])
        acc_batch = np.array([])
        start_time = time.time()
        for j in range(len(X_train)//BATCH_SIZE):
            print(f"Batch {j}", end="\r")
            X_batch = X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            y_batch = y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            LeNet5_numpy.fit_step(X_batch, y_batch, curr_step=epoch)
            loss_batch = np.append(loss_batch, LeNet5_numpy.loss)
            acc_batch = np.append(acc_batch, LeNet5_numpy.accuracy)
        end_time = time.time()
        LeNet5_numpy.validate_step(X=X_test[:2000], y=y_test[:2000], curr_step=epoch)
        print(f"Epoch {epoch}, training time = {end_time-start_time} s:") 
        print(f"Loss = {np.mean(loss_batch)}, Accuracy = {np.mean(acc_batch)}")
        print(f"Validation loss = {LeNet5_numpy.validation_loss}, Validation accuracy = {LeNet5_numpy.validation_accuracy}")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--use_proposed",
                        help="Train MLP using proposed method", action='store_false', default=True)
    args = parser.parse_args()
    main(use_proposed=args.use_proposed)