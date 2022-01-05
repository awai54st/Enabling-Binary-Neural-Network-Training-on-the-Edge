import tensorflow.keras.datasets.cifar10 as cifar10
import numpy as np
import matplotlib.pyplot as plt
import time
import IPython
import sys
import os
sys.path.insert(0, os.path.abspath("../"))

from core.label_utils import label_binerizer
from core.utils.training_utils import ImageDataGenerator
#del sys.modules["core.binary_models.layers"]
from core.numpy_models.full_precision_models.layers import Flatten, Softmax
from core.losses import CrossEntropy
from core.optimizers import GradientDescent, Adam
from core.label_utils import label_binerizer
from core.numpy_models.full_precision_models.pooling_layers import MaxPooling

from core.numpy_models.xnor_models.bit_packed_layers import XNorDense, XNorConv2D, BatchNorm
from core.numpy_models.binary_models.activation_layers import BinaryActivation
from core.model import Sequential

BATCH_NORM_MOMENTUM=0.9
BATCH_SIZE = 100
LR_ADAM=np.float32(1e-3) # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
DTYPE = np.float32


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = ((X_train.astype(DTYPE)*np.float32(2)-np.float32(255))/np.float32(255))
y_train = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
X_test = ((X_test.astype(DTYPE)*np.float32(2)-np.float32(255))/np.float32(255))
y_test = label_binerizer(y_test.reshape(-1).astype(int), n_class=10)

idx_choice = list(range(len(X_test)))

print("Training samples: ", X_train.shape)
print("Testing samples: ", X_test.shape)


### Binary net
test_data_generator = ImageDataGenerator()

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
#optimizer_op = GradientDescent(lr=LR_SGD)
optimizer_op = Adam(lr=LR_ADAM)

BinaryNet_numpy = Sequential(model_layers)
BinaryNet_numpy.compile(loss_op, optimizer_op)

np.random.seed(0)
for epoch in range(20):
    loss_batch = np.array([])
    acc_batch = np.array([])
    start_time = time.time()
    
    X_train_mod, y_train_mod = test_data_generator.augment_images(X_train, y_train)
    #X_train_mod = X_train
    
    for j in range(len(X_train_mod)//BATCH_SIZE):
        print(f"Batch {j}", end="\r")
        X_batch = X_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        y_batch = y_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        BinaryNet_numpy.fit_step(X_batch, y_batch)
        loss_batch = np.append(loss_batch, BinaryNet_numpy.loss)
        acc_batch = np.append(acc_batch, BinaryNet_numpy.accuracy)
        if j%100 == 0:
            print(f"Loss = {np.mean(loss_batch):.2f}, Acc = {np.mean(acc_batch):.2f}, param: {BinaryNet_numpy._patience}, {optimizer_op.lr}")
    end_time = time.time()
    rnd_idx = np.random.choice(idx_choice, size=1000, replace=False)
    BinaryNet_numpy.validate_step(X=X_test[rnd_idx], y=y_test[rnd_idx], curr_step=epoch)
    print(f"Epoch {epoch}, training time = {end_time-start_time} s:") 
    print(f"Loss = {np.mean(loss_batch)}, Accuracy = {np.mean(acc_batch)}")
    print(f"Validation loss = {BinaryNet_numpy.validation_loss}, Validation accuracy = {BinaryNet_numpy.validation_accuracy}")
