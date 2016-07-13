
from base_autoencoder import base_autoencoder
import numpy as np

class autoencoder(base_autoencoder):
    
    def __init__(self, layer_units, weights=None, bias=False, act_func='sigmoid', loss_type='cross_entropy', seed=12):
        base_autoencoder.__init__(self, layer_units, weights, bias, act_func, loss_type)
        self.init_weights(seed)
    
    def loss(self, X, reg=0.001): 
        weights = self.weights
        bias = self.bias
        
        W  = weights['W']
        b0 =0; b1 =0
        if bias == True:
            b0 = weights['b0']
            b1 = weights['b1']
            
        hid     = [X.dot(W)+b0, X.dot(W)] [bias == True]
        hid_act = self.ac_func(hid, self.act_func)
        pre     = [hid_act.dot(W.transpose())+b1, hid_act.dot(W.transpose())] [bias == True]
        X_pred  = self.ac_func(pre, self.act_func)
        
        loss = self.loss_func(X, X_pred)
        loss += reg*np.sum(W*W) # Regularization Term
        
        err_pred     = self.loss_func_deriv(X, X_pred)
        err_pred_act = err_pred*self.ac_func_deriv(X_pred, self.act_func)  
        err_W1       = err_pred_act.dot(W)
        
        err_hid_act = err_W1*self.ac_func_deriv(hid_act, self.act_func)
        err_W0      = err_hid_act.dot(W.transpose())
        
        dW  = hid_act.transpose().dot(err_pred_act).transpose()
        dW += X.transpose().dot(err_hid_act) 
        dW += reg*W
                
        if bias == True:
            db1  = np.sum(err_pred_act, axis=0)          
            db0  = np.sum(err_hid_act, axis=0)
             
        grads = {}   
        grads['W'] = dW
        if self.bias == True:
            grads['b0'] = db0
            grads['b1'] = db1
        
        return loss, grads
        
    
    