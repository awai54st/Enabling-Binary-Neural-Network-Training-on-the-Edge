import os
if os.environ.get("USE_CUPY"):
    import cupy as np
else:
    import numpy as np

class Sequential:
    def __init__(self, layers = []):
        self.layers = layers
        
        self._loss = None
        self._accuracy = None
        self._prev_acc = 0
        self._patience = 0
        
    def compile(self, losses_op, optimizers_op):
        self.losses_op = losses_op
        self.optimizers_op = optimizers_op
        
    def predict(self, input, training=False):
        x = input.copy()
        for idx, layer in enumerate(self.layers):
            assert not np.isnan(x).any()
            assert not np.isinf(x).any()
            x = layer.forward(x, training=training)
            if idx == (len(self.layers)-2):
                self.logits = x
            
        return x
        
    @property
    def loss(self):
        return np.mean(self._loss)
    
    @property
    def accuracy(self):
        return np.mean(self._accuracy)
        
    @property
    def validation_loss(self):
        return np.mean(self._validation_loss)
    
    @property
    def validation_accuracy(self):
        return np.mean(self._validation_accuracy)
    
    
    def backprop(self, initial_gradient):
        x = initial_gradient.copy()
        for layer in self.layers[::-1]:
            assert not np.isnan(x).any()
            assert not np.isinf(x).any()
            x = layer.backprop(x)
            """print(f"Min, max grad ({layer}): {np.min(x)}, {np.max(x)}")
            
            if layer.weights is None:
                continue
            print(f"Minimum weights ({layer}): {np.min(layer.weights[0])}, {np.max(layer.weights[0])}")
            print(f"NaN weights ({layer}): {np.isnan(layer.weights[0]).any()}")"""

    def reduce_lr_on_plateau(self, acc):
        if self._prev_acc < acc:
            self._prev_acc = acc
            self._patience = 0
        else:
            self._patience += 1
        
    def fit_step(self, X, y, curr_step=0, batch_size=500):
        predictions = self.predict(X, training=True)
        #print(f"NaN predictions: {np.isnan(predictions).any()}")
        self._loss = self.losses_op.forward(self.logits, y)
        # self._accuracy = (y == predictions).all(axis=-1).sum()
        self._accuracy = (y.argmax(axis=-1) == predictions.argmax(axis=-1))
        # print(predictions.argmax(axis=-1))
        initial_gradient = self.losses_op.backprop()
        self.backprop(initial_gradient)
        self.optimizers_op.update(self.layers, curr_step)
        
        self.reduce_lr_on_plateau(self.accuracy)
        if self._patience == 50*batch_size:
            self._patience = 0
            self.optimizers_op.params["lr"] = np.float32(0.5) * self.optimizers_op.params["lr"] 

        
    def validate_step(self, X, y, curr_step):
        predictions = self.predict(X, training=False)
        #print(f"NaN predictions: {np.isnan(predictions).any()}")
        self._validation_loss = self.losses_op.forward(self.logits, y)
        # self._accuracy = (y == predictions).all(axis=-1).sum()
        self._validation_accuracy = (y.argmax(axis=-1) == predictions.argmax(axis=-1))
        # print(predictions.argmax(axis=-1))