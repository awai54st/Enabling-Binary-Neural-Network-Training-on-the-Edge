import tensorflow.keras.datasets.cifar10 as cifar10
import time
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Activation, BatchNormalization, MaxPooling2D

from core.label_utils import label_binerizer
from core.keras_models.keras_binary_models.binary_tf_model import BinaryActivationTf, DenseTf, Conv2DTf, l1_batch_norm_mod_conv, l1_batch_norm_mod_dense

def binary_net_training_pipeline(epochs=1, batch_size=100):
    BATCH_NORM_MOMENTUM=0.9
    BATCH_SIZE = batch_size
    BATCH_NORM_EPS=1e-4
    LR_ADAM=np.float32(1e-3) # from paper pg 18: https://openreview.net/pdf?id=pwwVuSICBgt
    DTYPE = np.float32
    MULTIPLE = 1

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = ((X_train.astype(DTYPE)*np.float32(2)-np.float32(255))/np.float32(255))
    y_train = label_binerizer(y_train.reshape(-1).astype(int), n_class=10)
    X_train = X_train[:MULTIPLE*BATCH_SIZE]
    y_train = y_train[:MULTIPLE*BATCH_SIZE]
    X_test = None
    y_test = None

    print("Training samples: ", X_train.shape)


    model = Sequential(
        [
            Input(shape=(32, 32, 3)),
            Conv2DTf(128, (3,3), padding="same"),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),
            
            Conv2DTf(128, (3,3), padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),
            
            
            Conv2DTf(256, (3,3), padding="same"),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),
            
            Conv2DTf(256, (3,3), padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),
            
            
            Conv2DTf(512, (3,3), padding="same"),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),
            
            Conv2DTf(512, (3,3), padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),
            
            Flatten(),

            DenseTf(1024, activation=None, name="layer1"),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),

            DenseTf(1024, activation=None, name="layer2"),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            BinaryActivationTf(),

            DenseTf(10, name="layer_output"),
            BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPS),
            Activation('softmax')
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(lr=LR_ADAM),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE)