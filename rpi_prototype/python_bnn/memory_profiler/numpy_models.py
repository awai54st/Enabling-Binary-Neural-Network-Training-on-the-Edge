import tensorflow.keras.datasets.cifar10 as cifar10
import tensorflow.keras.datasets.mnist as mnist
import time
import numpy as np

from core.label_utils import label_binerizer
from core.utils.training_utils import ImageDataGenerator

from core.numpy_models.full_precision_models.layers import Flatten, Softmax
from core.losses import CrossEntropy
from core.optimizers import GradientDescent, Adam
from core.label_utils import label_binerizer
from core.numpy_models.full_precision_models.pooling_layers import MaxPooling
#from core.cython.max_pooling import MaxPooling

from core.numpy_models.xnor_models.bit_packed_layers import XNorDense, XNorConv2D, BatchNorm

from core.numpy_models.binary_models.layers import BinaryDense, BinaryConv2D, BatchNorm as VBatchNorm
from core.numpy_models.binary_models.activation_layers import BinaryActivation
from core.model import Sequential



def xNOR_mlp_training_pipeline(batch_size=100, epochs=1, DTYPE=np.float16):
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = batch_size
    LR_ADAM=1e-3 # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    DTYPE = np.float32
    MULTIPLE = 1
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train.astype(DTYPE), -1)/255
    y_train = label_binerizer(y_train.astype(int), n_class=10)
    X_test = None
    y_test = None
    X_train = X_train[:MULTIPLE*BATCH_SIZE]
    y_train = y_train[:MULTIPLE*BATCH_SIZE]
    print("Training samples: ", X_train.shape)
    
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
        XNorDense(256, dtype=DTYPE, random_seed=4),
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

    for epoch in range(epochs):
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
        print(f"Epoch {epoch}, training time = {end_time-start_time} s:") 
        print(f"Loss = {np.mean(loss_batch)}, Accuracy = {np.mean(acc_batch)}")



def xNOR_binary_net_training_pipeline(batch_size=100, epochs=1, DTYPE=np.float16):
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = batch_size
    LR_ADAM=np.float32(1e-3) # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    MULTIPLE = 1
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = ((X_train.astype(DTYPE)*DTYPE(2)-DTYPE(255))/DTYPE(255))
    y_train = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
    X_test = None
    y_test = None
    
    #X_test = ((X_train.astype(DTYPE)*np.float32(2)-np.float32(255))/np.float32(255))
    #y_test = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
    X_train = X_train[:MULTIPLE*BATCH_SIZE]
    y_train = y_train[:MULTIPLE*BATCH_SIZE]

    print("Training samples: ", X_train.shape)

    ### Binary net
    test_data_generator = ImageDataGenerator()

    model_layers = [
        # 128
        XNorConv2D(128, (3,3), padding="same", dtype=np.float32, random_seed=0, first_layer=True),
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
    optimizer_op = Adam(lr=LR_ADAM, dtype=DTYPE)

    BinaryNet_numpy = Sequential(model_layers)
    BinaryNet_numpy.compile(loss_op, optimizer_op)

    np.random.seed(0)
    for epoch in range(epochs):
        loss_batch = np.array([])
        acc_batch = np.array([])
        start_time = time.time()

        X_train_mod, y_train_mod = test_data_generator.augment_images(X_train, y_train)

        for j in range(MULTIPLE):
            print(f"Batch {j}", end="\r")
            X_batch = X_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            y_batch = y_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            BinaryNet_numpy.fit_step(X_batch, y_batch)
            loss_batch = np.append(loss_batch, BinaryNet_numpy.loss)
            acc_batch = np.append(acc_batch, BinaryNet_numpy.accuracy)
        end_time = time.time()
        print(f"Epoch {epoch}, training time = {end_time-start_time} s:") 
        print(f"Loss = {np.mean(loss_batch)}, Accuracy = {np.mean(acc_batch)}")


def vanilla_mlp_training_pipeline(batch_size=100, epochs=1, DTYPE=np.float32):
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = batch_size
    LR_ADAM=1e-3 # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    DTYPE = np.float32
    MULTIPLE = 1
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train.astype(DTYPE), -1)/255
    y_train = label_binerizer(y_train.astype(int), n_class=10)
    X_test = None
    y_test = None
    X_train = X_train[:MULTIPLE*BATCH_SIZE]
    y_train = y_train[:MULTIPLE*BATCH_SIZE]
    print("Training samples: ", X_train.shape)
    
    model_layers = [
        Flatten(), 
        BinaryDense(256, dtype=np.float32, random_seed=0),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(256, dtype=DTYPE, random_seed=1),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(256, dtype=DTYPE, random_seed=2),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(256, dtype=DTYPE, random_seed=4),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        BinaryDense(10, dtype=DTYPE, random_seed=4),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        Softmax(dtype=DTYPE)
    ]

    loss_op = CrossEntropy(dtype=DTYPE)
    optimizer_op = Adam(lr=LR_ADAM, dtype=DTYPE)

    LeNet5_numpy = Sequential(model_layers)
    LeNet5_numpy.compile(loss_op, optimizer_op)

    for epoch in range(epochs):
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
        print(f"Epoch {epoch}, training time = {end_time-start_time} s:") 
        print(f"Loss = {np.mean(loss_batch)}, Accuracy = {np.mean(acc_batch)}")
        
def vanilla_binary_net_training_pipeline(batch_size=100, epochs=1, DTYPE=np.float16):
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = batch_size
    LR_ADAM=np.float32(1e-3)
    MULTIPLE = 1
    

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = ((X_train.astype(DTYPE)*DTYPE(2)-DTYPE(255))/DTYPE(255))
    y_train = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
    X_test = None
    y_test = None
    
    #X_test = ((X_train.astype(DTYPE)*DTYPE(2)-DTYPE(255))/DTYPE(255))
    #y_test = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
    X_train = X_train[:MULTIPLE*BATCH_SIZE]
    y_train = y_train[:MULTIPLE*BATCH_SIZE]
    
    
    print("Training samples: ", X_train.shape)
    
    ### Binary net
    test_data_generator = ImageDataGenerator()

    model_layers = [
        # 128
        BinaryConv2D(128, (3,3), padding="same", dtype=DTYPE, random_seed=0),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        BinaryConv2D(128, (3,3), padding="same", dtype=DTYPE, random_seed=1),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        # 256
        BinaryConv2D(256, (3,3), padding="same", dtype=DTYPE, random_seed=2),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        BinaryConv2D(256, (3,3), padding="same", dtype=DTYPE, random_seed=3),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        # 512
        BinaryConv2D(512, (3,3), padding="same", dtype=DTYPE, random_seed=4),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        BinaryConv2D(512, (3,3), padding="same", dtype=DTYPE, random_seed=5),
        MaxPooling((2,2), (2,2), dtype=DTYPE),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),

        Flatten(dtype=DTYPE), 
        BinaryDense(1024, dtype=DTYPE, random_seed=0),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        
        BinaryDense(1024, dtype=DTYPE, random_seed=1),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        BinaryActivation(dtype=DTYPE),
        
        BinaryDense(10, dtype=DTYPE, random_seed=2),
        VBatchNorm(BATCH_NORM_MOMENTUM, dtype=DTYPE),
        Softmax(dtype=DTYPE)
    ]

    loss_op = CrossEntropy(dtype=DTYPE)
    optimizer_op = Adam(lr=LR_ADAM)

    BinaryNet_numpy = Sequential(model_layers)
    BinaryNet_numpy.compile(loss_op, optimizer_op)

    np.random.seed(0)
    for epoch in range(epochs):
        loss_batch = np.array([])
        acc_batch = np.array([])
        start_time = time.time()

        X_train_mod, y_train_mod = test_data_generator.augment_images(X_train, y_train)

        for j in range(MULTIPLE):
            print(f"Batch {j}", end="\r")
            X_batch = X_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            y_batch = y_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            BinaryNet_numpy.fit_step(X_batch, y_batch)
            loss_batch = np.append(loss_batch, BinaryNet_numpy.loss)
            acc_batch = np.append(acc_batch, BinaryNet_numpy.accuracy)
        end_time = time.time()
        print(f"Epoch {epoch}, training time = {end_time-start_time} s:") 
        print(f"Loss = {np.mean(loss_batch)}, Accuracy = {np.mean(acc_batch)}")
        

def xNOR_binary_net_disect_training_pipeline(epochs=1, batch_size=100, DTYPE=np.float16):
    if DTYPE ==np.float16:
        EPS = 1e-7
    else:
        EPS = 1e-45
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = batch_size
    LR_ADAM=np.float32(1e-3) # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = ((X_train.astype(DTYPE)*DTYPE(2)-DTYPE(255))/DTYPE(255))
    y_train = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
    X_test = None
    y_test = None
    
    #X_test = ((X_train.astype(DTYPE)*np.float32(2)-np.float32(255))/np.float32(255))
    #y_test = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
    X_train = X_train[:2*BATCH_SIZE]
    y_train = y_train[:2*BATCH_SIZE]

    print("Training samples: ", X_train.shape)

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
    optimizer_op = Adam(lr=LR_ADAM, dtype=DTYPE)

    BinaryNet_numpy = Sequential(model_layers)
    BinaryNet_numpy.compile(loss_op, optimizer_op)

    np.random.seed(0)
    for epoch in range(epochs):
        loss_batch = np.array([])
        acc_batch = np.array([])
        start_time = time.time()

        X_train_mod, y_train_mod = test_data_generator.augment_images(X_train, y_train)

        for j in range(len(X_train)//BATCH_SIZE):
            print(f"Batch {j}", end="\r")
            X_batch = X_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            y_batch = y_train_mod[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            
            x1 = BinaryNet_numpy.layers[0].forward(X_batch)
            x2 = BinaryNet_numpy.layers[1].forward(x1)
            x1 = None
            x3 = BinaryNet_numpy.layers[2].forward(x2)
            x2 = None
            x4 = BinaryNet_numpy.layers[3].forward(x3)
            x3 = None
            x5 = BinaryNet_numpy.layers[4].forward(x4)
            x4 = None
            x6 = BinaryNet_numpy.layers[5].forward(x5)
            x5 = None
            x7 = BinaryNet_numpy.layers[6].forward(x6)
            x6 = None
            x8 = BinaryNet_numpy.layers[7].forward(x7)
            x7 = None
            x9 = BinaryNet_numpy.layers[8].forward(x8)
            x8 = None
            x10 = BinaryNet_numpy.layers[9].forward(x9)
            x9 = None
            x11 = BinaryNet_numpy.layers[10].forward(x10)
            x10 = None
            x12 = BinaryNet_numpy.layers[11].forward(x11)
            x11 = None
            x13 = BinaryNet_numpy.layers[12].forward(x12)
            x12 = None
            x14 = BinaryNet_numpy.layers[13].forward(x13)
            x13 = None
            x15 = BinaryNet_numpy.layers[14].forward(x14)
            x14 = None
            x16 = BinaryNet_numpy.layers[15].forward(x15)
            x15 = None
            x17 = BinaryNet_numpy.layers[16].forward(x16)
            x16 = None
            x18 = BinaryNet_numpy.layers[17].forward(x17)
            x17 = None
            x19 = BinaryNet_numpy.layers[18].forward(x18)
            x18 = None
            x20 = BinaryNet_numpy.layers[19].forward(x19)
            x19 = None
            x21 = BinaryNet_numpy.layers[20].forward(x20)
            x20 = None
            x22 = BinaryNet_numpy.layers[21].forward(x21)
            x21 = None
            x23 = BinaryNet_numpy.layers[22].forward(x22)
            x22 = None
            x24 = BinaryNet_numpy.layers[23].forward(x23)
            x23 = None
            x25 = BinaryNet_numpy.layers[24].forward(x24)
            x24 = None
            x26 = BinaryNet_numpy.layers[25].forward(x25)
            x25 = None
            x27 = BinaryNet_numpy.layers[26].forward(x26)
            x26 = None
            x28 = BinaryNet_numpy.layers[27].forward(x27)
            x27 = None
            x29 = BinaryNet_numpy.layers[28].forward(x28)
            x28 = None
            x30 = BinaryNet_numpy.layers[29].forward(x29)
            x29 = None
            y = BinaryNet_numpy.layers[30].forward(x30)
            x30 = None
            
            loss = BinaryNet_numpy.losses_op.forward(y, y_batch)
        
            initial_loss = BinaryNet_numpy.losses_op.backprop()
            # backward
            x1 = BinaryNet_numpy.layers[30].backprop(initial_loss)
            x2 = BinaryNet_numpy.layers[29].backprop(x1)
            x1 = None
            x3 = BinaryNet_numpy.layers[28].backprop(x2)
            x2 = None
            x4 = BinaryNet_numpy.layers[27].backprop(x3)
            x3 = None
            x5 = BinaryNet_numpy.layers[26].backprop(x4)
            x4 = None
            x6 = BinaryNet_numpy.layers[25].backprop(x5)
            x5 = None
            x7 = BinaryNet_numpy.layers[24].backprop(x6)
            x6 = None
            x8 = BinaryNet_numpy.layers[23].backprop(x7)
            x7 = None
            x9 = BinaryNet_numpy.layers[22].backprop(x8)
            x8 = None
            x10 = BinaryNet_numpy.layers[21].backprop(x9)
            x9 = None
            x11 = BinaryNet_numpy.layers[20].backprop(x10)
            x10 = None
            x12 = BinaryNet_numpy.layers[19].backprop(x11)
            x11 = None
            x13 = BinaryNet_numpy.layers[18].backprop(x12)
            x12 = None
            x14 = BinaryNet_numpy.layers[17].backprop(x13)
            x13 = None
            x15 = BinaryNet_numpy.layers[16].backprop(x14)
            x14 = None
            x16 = BinaryNet_numpy.layers[15].backprop(x15)
            x15 = None
            x17 = BinaryNet_numpy.layers[14].backprop(x16)
            x16 = None
            x18 = BinaryNet_numpy.layers[13].backprop(x17)
            x17 = None
            x19 = BinaryNet_numpy.layers[12].backprop(x18)
            x18 = None
            x20 = BinaryNet_numpy.layers[11].backprop(x19)
            x19 = None
            x21 = BinaryNet_numpy.layers[10].backprop(x20)
            x20 = None
            x22 = BinaryNet_numpy.layers[9].backprop(x21)
            x21 = None
            x23 = BinaryNet_numpy.layers[8].backprop(x22)
            x22 = None
            x24 = BinaryNet_numpy.layers[7].backprop(x23)
            x23 = None
            x25 = BinaryNet_numpy.layers[6].backprop(x24)
            x24 = None
            x26 = BinaryNet_numpy.layers[5].backprop(x25)
            x25 = None
            x27 = BinaryNet_numpy.layers[4].backprop(x26)
            x26 = None
            x28 = BinaryNet_numpy.layers[3].backprop(x27)
            x27 = None
            x29 = BinaryNet_numpy.layers[2].backprop(x28)
            x28 = None
            x30 = BinaryNet_numpy.layers[1].backprop(x29)
            x29 = None
            y = BinaryNet_numpy.layers[0].backprop(x30)
            x30 = None
        
            BinaryNet_numpy.optimizers_op.update(BinaryNet_numpy.layers, j)
        
            #loss_batch = np.append(loss_batch, BinaryNet_numpy.loss)
            #acc_batch = np.append(acc_batch, BinaryNet_numpy.accuracy)
        end_time = time.time()
        #print(f"Epoch {epoch}, training time = {end_time-start_time} s:") 
        #print(f"Loss = {np.mean(loss_batch)}, Accuracy = {np.mean(acc_batch)}")
        

