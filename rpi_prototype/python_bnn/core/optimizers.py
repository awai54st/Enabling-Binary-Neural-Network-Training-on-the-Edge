import numpy as np


class GradientDescent:
    def __init__(self, lr):
        """
        Args:
            lr (float): learning rate
        """
        self.lr = lr
        
    def update(self, layers, curr_step):
        """
        Args:
            layers (list): list of layers in the model
        """
        for layer in layers:
            weights = layer.weights
            gradients = layer.gradients
            
            #print(f"Layers: {layer}")
            #print(gradients)
            if (weights is None) or (gradients is None):
                continue
            '''if weights is None:
                raise Exception("NaN weights")
            if gradients is None:
                raise Exception("NaN gradients")'''
                       
            #print(f"weights: {weights[0].dtype}")
            #print(f"bias: {weights[1].dtype}")
            #print(f"weights gradients: {gradients[0].dtype}")
            #print(f"bias gradients: {gradients[1].dtype}")
            
            w, b = weights
            dw, db = gradients
            layer.set_weights(
                w = w - self.lr*dw,
                b = b - self.lr*db
            )
            
            """
            layer.set_weights(
                *[_x*_g for _x, _g in zip(weights, gradients)]
            )"""
            
            
class AdamBase:
    # https://arxiv.org/pdf/1412.6980.pdf
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
    def __init__(self):
        """
        Args:
            lr: step size (learning rate)
            beta1 = exponential decay rates for moment estimates
            beta2 = exponential decay rates for moment estimates
        """
        # initialize 1st moment vector, 2nd moment vector for w and b
        self.m_adam_arr = []
        self.v_adam_arr = []
        
            
    def calculate_corrected_gradient(self, grads, params):
        ONE = np.float32(1)
        
        if not self.m_adam_arr:
            self.m_adam_arr = [params["dtype"](0)]*len(grads)
            
        if not self.v_adam_arr:
            self.v_adam_arr = [params["dtype"](0)]*len(grads)
            
        # lr
        lr = params["lr"] * np.sqrt(ONE-params["beta2"]**params["curr_step"])/(ONE-params["beta1"]**params["curr_step"])
        lr = lr.astype(np.float32)
        for idx, grad in enumerate(grads):
            grad = grad.astype(np.float32)
            # m(t) = beta1*m(t-1)+(1-beta1)*grad
            self.m_adam_arr[idx] = params["beta1"]*self.m_adam_arr[idx] + (ONE-params["beta1"])*(grad)
            
            # v(t) = beta2*v(t-1)+(1-beta2)*grad^2
            self.v_adam_arr[idx] = params["beta2"]*self.v_adam_arr[idx] + (ONE-params["beta2"])*(grad*grad)
        
        #self.m_db = self.beta1*self.m_db + (ONE-self.beta1)*db
        #self.v_db = self.beta2*self.v_db + (ONE-self.beta2)*(db**TWO)
        
        if not params["amsgrad"]:
            return self.m_adam_arr, self.v_adam_arr, lr
    
        # bias correction
        # amsgrad
        m_adam_corr_arr = []
        v_adam_corr_arr = []
        for m_adam, v_adam in zip(self.m_adam_arr, self.v_adam_arr):
            # m_corr = m/(1-beta1^curr_step)
            m_adam_corr_arr.append(m_adam/(ONE-params["beta1"]**params["curr_step"]))

            # v_corr = v/(1-beta2^curr_step)
            v_adam_corr_arr.append(v_adam/(ONE-params["beta2"]**params["curr_step"]))

        #m_db_corr = self.m_db/(1-self.beta1**curr_step)
        #v_db_corr = self.v_db/(1-self.beta2**curr_step)

        return m_adam_corr_arr, v_adam_corr_arr, lr
    
    
class Adam:
    # https://arxiv.org/pdf/1412.6980.pdf
    #https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, dtype=np.float32, amsgrad=False):
        """
        Args:
            lr: step size (learning rate)
            beta1 = exponential decay rates for moment estimates
            beta2 = exponential decay rates for moment estimates
        """
        self.params = {
            "dtype": dtype,
            "beta1": dtype(beta1), 
            "beta2": dtype(beta2),
            "epsilon": dtype(epsilon),
            "lr": dtype(lr),
            "amsgrad": amsgrad,
            "curr_step": dtype(0)
        }
        
        self.built = False
        self.adam_dict = {}
        
    def update(self, layers, curr_step=None):
        self.params["curr_step"] +=1
        for layer in layers:
            #print(gradients)
            if (layer.weights is None) or (layer.gradients is None):
                continue
                
            if not self.built:
                self.adam_dict[layer] = AdamBase()
            '''if weights is None:
                raise Exception("NaN weights")
            if gradients is None:
                raise Exception("NaN gradients")'''
            
            m_adams, v_adams, lr = self.adam_dict[layer].calculate_corrected_gradient(
                grads=layer.gradients, params=self.params)
            
            # update weights and biases
            # theta(t) = theta(t-1) - alpha*m_corr/(sqrt(v_corr) + eps)
            """layer.set_weights(
                w = w - self.lr*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon)),
                b = b - self.lr*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
            )"""
            layer.set_weights(
                [w-lr*m_adam/(np.sqrt(v_adam)+self.params["epsilon"]) for w, m_adam, v_adam in zip(layer.weights, m_adams, v_adams)]
            )
        self.built = True
        return self.params["curr_step"]
        