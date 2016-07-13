#! /usr/bin/python

import numpy as np

def gaussiannoise(rate=0.5, sd=0.5):
    def func(x):
        noisy_x = np.copy(x)
        mask = (np.random.uniform(0,1, x.shape)<rate).astype("i4")
        noisy_x += mask * np.random.normal(0, sd, x.shape)
        return noisy_x
    
    return func

class base_autoencoder:
    
    def __init__(self, layer_units, weights=None, bias=False, act_func='sigmoid', loss_type = 'cross_entropy'):
        self.weights = weights
        self.layer_units = layer_units
        self.bias = bias
        self.act_func = act_func
        self.loss_type = loss_type
        
    def init_weights(self, seed=0):

        layer_units = self.layer_units
        n_layers = len(layer_units)
        assert n_layers == 3

        np.random.seed(seed)

        # Since network consists of only three layer
        # Xavier parameter initialization as in Caffe.
        r  = np.sqrt(6) / np.sqrt(layer_units[1] + layer_units[0])
      
        weights = {}
        weights['W'] = np.random.random((layer_units[0], layer_units[1])) * 2.0 * r - r    
        if self.bias == True:
            weights['b0'] = np.zeros(layer_units[1])
            weights['b1'] = np.zeros(layer_units[2])
        
        self.weights = weights

        return self.weights
    
    def relu(self, x):
        return x*(x>0)
    
    def relu_deriv(self, x):
        return x*(x>0)
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1.0- x)
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        
    def tanh_deriv(self, x):
        return 1.0 - x*x

    def ac_func(self, x, func_name = 'sigmoid' ):
        if func_name == 'sigmoid':
            return self.sigmoid(x)
        elif func_name == 'tanh':
            return self.tanh(x)
        elif func_name == 'relu':
            return self.relu(x)
        else:
            raise func_name + " Not defined"

    def ac_func_deriv(self, x, func_name = 'sigmoid' ):
        if func_name == 'sigmoid':
            return self.sigmoid_deriv(x)
        elif func_name == 'tanh':
            return self.tanh_deriv(x)
        elif func_name == 'relu':
            return self.relu_deriv(x)
        else:
            raise func_name + " Not defined"
             
    def predict(self, X):
        weights = self.weights
        bias = self.bias
        
        W  = weights['W']
        b0=0; b1=0
        if bias == True:
            b0 = weights['b0']
            b1 = weights['b1']
        
        hid     = [X.dot(W)+b0, X.dot(W)] [bias == True]
        hid_act = self.ac_func(hid, self.act_func)
        pre     = [hid_act.dot(W.transpose())+b1, hid_act.dot(W.transpose())] [bias == True]
        X_pred  = self.ac_func(pre, self.act_func)
        
        return X_pred
    
    def loss(self, X, reg=0.001):
        raise NotImplementedError
    
    def loss_with_noise(self, X, X_noisy, reg=0.001):
        raise NotImplementedError
    
    def euclidean_loss(self, X, X_pred):
        ntrain  = X.shape[0]
        loss  = 0.5*np.sum( (X_pred-X)*(X_pred-X) ) / ntrain
        return loss
    
    def euclidean_loss_deriv(self, X, X_pred):
        ntrain = X.shape[0]
        deriv = (X_pred-X) / ntrain         
        return deriv
                
    def cross_entropy_loss(self, X, X_pred):
        ntrain  = X.shape[0]
        loss = - np.sum( X*np.log2(X_pred) + (1 - X)*np.log2(1 - X_pred) ) / ntrain
        return loss         
    
    def cross_entropy_loss_deriv(self, X, X_pred):
        ntrain  = X.shape[0]
        deriv = ( (X_pred-X)/(X_pred*(1-X_pred)) ) / ntrain
        return deriv
    
    def loss_func(self, X, X_pred):
        if self.loss_type == 'euclidean':
            return self.euclidean_loss(X, X_pred)
        elif self.loss_type == 'cross_entropy':
            return self.cross_entropy_loss(X, X_pred)
        else: 
            raise self.loss_type + " Not defined"
        
    def loss_func_deriv(self, X, X_pred):
        if self.loss_type == 'euclidean':
            return self.euclidean_loss_deriv(X, X_pred)
        elif self.loss_type == 'cross_entropy':
            return self.cross_entropy_loss_deriv(X, X_pred)
        else: 
            raise self.loss_type + " Not defined"

    def train_with_SGD(self, X, learning_rate=1e-3, learning_rate_decay=0.95, reg=3e-3, 
                       num_iters=1000, batchsize=128, mu=0.9):
        
        ntrain = X.shape[0]
        loss_history = []
        
        velocity = {}
        velocity['W'] = np.zeros((self.layer_units[0], self.layer_units[1]))
        
        if self.bias == True:
            velocity['b0'] = np.zeros(self.layer_units[1])
            velocity['b1'] = np.zeros(self.layer_units[2])
            
        for it in xrange(num_iters):
            
            batch_indicies = np.random.choice(ntrain, batchsize, replace = False)
            X_batch = X[batch_indicies]
            
            loss, grads = self.loss(X_batch, reg)
            loss_history.append(loss)

            W  = self.weights['W']
            wgrad = learning_rate*grads['W'] + mu*velocity['W']
            W -= wgrad
            velocity['W'] = wgrad

            if self.bias == True:
                b0 = self.weights['b0']
                b1 = self.weights['b1']
                
                b1grad = learning_rate*grads['b1'] + mu*velocity['b1'] 
                b1 -= b1grad 
                velocity['b1'] = b1grad
            
                b0grad = learning_rate*grads['b0'] + mu*velocity['b0'] 
                b0 -= b0grad
                velocity['b0'] = b0grad
                
            if it % 50 == 0:
                print 'SGD: iteration %d / %d: loss %f' % (it, num_iters, loss)
                    
            if it % 200 == 0:
                learning_rate *= learning_rate_decay
                
        return loss_history
                
                
    def train_with_SGD_with_noise(self, X, noise=gaussiannoise(sd=0.5), learning_rate=1e-3, 
                                  learning_rate_decay=0.95, reg=3e-3, 
                                  num_iters=1000, batchsize=128, mu=0.9):
        
        ntrain = X.shape[0]
        loss_history = []
        
        velocity = {}
        velocity['W'] = np.zeros((self.layer_units[0], self.layer_units[1]))
        
        if self.bias == True:
            velocity['b0'] = np.zeros(self.layer_units[1])
            velocity['b1'] = np.zeros(self.layer_units[2])
            
        for it in xrange(num_iters):
            
            batch_indicies = np.random.choice(ntrain, batchsize, replace = False)
            X_batch       = X[batch_indicies]
            X_batch_noise = noise(X_batch)
            
            loss, grads = self.loss_with_noise(X_batch, X_batch_noise, reg)
            loss_history.append(loss)

            W  = self.weights['W']
            wgrad = learning_rate*grads['W'] + mu*velocity['W']
            W -= wgrad
            velocity['W'] = wgrad

            if self.bias == True:
                b0 = self.weights['b0']
                b1 = self.weights['b1']
                
                b1grad = learning_rate*grads['b1'] + mu*velocity['b1'] 
                b1 -= b1grad 
                velocity['b1'] = b1grad
            
                b0grad = learning_rate*grads['b0'] + mu*velocity['b0'] 
                b0 -= b0grad
                velocity['b0'] = b0grad
                
            if it % 50 == 0:
                print 'SGD: iteration %d / %d: loss %f' % (it, num_iters, loss)
                
            if it % 200 == 0:
                learning_rate *= learning_rate_decay
                
        return loss_history