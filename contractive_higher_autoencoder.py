
from base_autoencoder import base_autoencoder
import numpy as np

class contractive_higher_autoencoder(base_autoencoder):
    
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
                
        ntrain  = X.shape[0]
        ninput  = X.shape[1]
        nhidden = W.shape[1]
            
        hid     = [X.dot(W)+b0, X.dot(W)] [bias == True]
        hid_act = self.ac_func(hid, self.act_func)
        pre     = [hid_act.dot(W.transpose())+b1, hid_act.dot(W.transpose())] [bias == True]
        X_pred  = self.ac_func(pre, self.act_func)
        
        loss = self.loss_func(X, X_pred)
        act_der = np.reshape(hid_act*(1.0-hid_act)*(1-2.0*hid_act), (ntrain, 1, nhidden))
        loss_jacobian = act_der*np.reshape(W, (1, ninput, nhidden))
        loss += reg*np.sum( loss_jacobian*loss_jacobian ) /ntrain  
        
        err_pred     = self.loss_func_deriv(X, X_pred)
        err_pred_act = err_pred*self.ac_func_deriv(X_pred, self.act_func)  
        err_W1       = err_pred_act.dot(W)
        
        # This part 
        err_hid_act = err_W1*self.ac_func_deriv(hid_act, self.act_func) + reg*(hid_act*(1.0-hid_act)*(1.0-6.0*hid_act+6.0*hid_act*hid_act))*np.sum(W*W, axis=0)/ntrain
        err_W0      = err_hid_act.dot(W.transpose())
        
        dW  = hid_act.transpose().dot(err_pred_act).transpose()
        dW += X.transpose().dot(err_hid_act) 
        
        dW += reg*np.sum(act_der*act_der)*W/ntrain    
    
        if bias == True:
            db1  = np.sum(err_pred_act, axis=0)          
            db0  = np.sum(err_hid_act, axis=0)
             
        grads = {}   
        grads['W'] = dW
        if self.bias == True:
            grads['b0'] = db0
            grads['b1'] = db1
        
        return loss, grads
        
    
    