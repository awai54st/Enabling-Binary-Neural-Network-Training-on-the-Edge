import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import math
import keras
from keras.datasets import cifar10,mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
sys.path.insert(0, '..')
from binarization_utils import *
from model_architectures import get_model

dataset='binarynet'
Train=True
Evaluate=True
batch_size=100
epochs=1000

def load_svhn(path_to_dataset):
	import scipy.io as sio
	train=sio.loadmat(path_to_dataset+'/train.mat')
	test=sio.loadmat(path_to_dataset+'/test.mat')
	extra=sio.loadmat(path_to_dataset+'/extra.mat')
	X_train=np.transpose(train['X'],[3,0,1,2])
	y_train=train['y']-1

	X_test=np.transpose(test['X'],[3,0,1,2])
	y_test=test['y']-1

	X_extra=np.transpose(extra['X'],[3,0,1,2])
	y_extra=extra['y']-1

	X_train=np.concatenate((X_train,X_extra),axis=0)
	y_train=np.concatenate((y_train,y_extra),axis=0)

	return (X_train,y_train),(X_test,y_test)

if dataset=="MNIST":
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# convert class vectors to binary class matrices
	X_train = X_train.reshape(-1,784)
	X_test = X_test.reshape(-1,784)
	use_generator=False
elif dataset=="CIFAR-10" or dataset=="binarynet":
	use_generator=True
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
elif dataset=="SVHN" or dataset=="binarynet-svhn":
	use_generator=True
	(X_train, y_train), (X_test, y_test) = load_svhn('./svhn_data')
else:
	raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN, binarynet, binarynet-svhn].")

X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train /= 255
X_test /= 255
X_train=2*X_train-1
X_test=2*X_test-1


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.025
	drop = 0.5
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

if Train:
	if not(os.path.exists('models')):
		os.mkdir('models')
	if not(os.path.exists('models/'+dataset)):
		os.mkdir('models/'+dataset)
	for resid_levels in range(1):
		print 'training with', resid_levels,'levels'
		sess=K.get_session()
		model=get_model(dataset,resid_levels,batch_size)
		#model.summary()

		#gather all binary dense and binary convolution layers:
		binary_layers=[]
		for l in model.layers:
			if isinstance(l,binary_dense) or isinstance(l,binary_conv):
				binary_layers.append(l)

		#gather all residual binary activation layers:
		resid_bin_layers=[]
		for l in model.layers:
			if isinstance(l,binary_activation):
				resid_bin_layers.append(l)
		lr=0.001
		opt = keras.optimizers.Adam(lr=lr)#SGD(lr=lr,momentum=0.9,decay=1e-5)
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		# learning rate scheduler callback
		lrate = keras.callbacks.ReduceLROnPlateau(
			monitor='val_acc', factor=0.5, patience=50, verbose=0, mode='auto',
			min_delta=0.0001, cooldown=0, min_lr=0
		)
		cback=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True)
		if use_generator:
			if dataset=="CIFAR-10" or dataset=="binarynet":
				horizontal_flip=True
			if dataset=="SVHN" or dataset=="binarynet-svhn":
				horizontal_flip=False
			datagen = ImageDataGenerator(
				width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=horizontal_flip)  # randomly flip images
			if keras.__version__[0]=='2':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),steps_per_epoch=X_train.shape[0]/batch_size,
				nb_epoch=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[lrate, cback])
			if keras.__version__[0]=='1':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size), samples_per_epoch=X_train.shape[0], 
				nb_epoch=epochs, verbose=2,validation_data=(X_test,y_test),callbacks=[lrate, cback])

		else:
			if keras.__version__[0]=='2':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[lrate, cback])
			if keras.__version__[0]=='1':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,nb_epoch=epochs,callbacks=[lrate, cback])
		dic={'hard':history.history}
		foo=open('models/'+dataset+'/history_'+str(resid_levels)+'_residuals.pkl','wb')
		pickle.dump(dic,foo)
		foo.close()

if Evaluate:
	for resid_levels in range(1):
		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		model=get_model(dataset,resid_levels,batch_size)
		model.load_weights(weights_path)
		opt = keras.optimizers.Adam()
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		score=model.evaluate(X_test,y_test,verbose=0, batch_size=batch_size)
		print "with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1])



