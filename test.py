
import numpy as np
import matplotlib.pyplot as plt
import time

# Local 
from data_utils import load_mnist
from base_autoencoder import gaussiannoise
from autoencoder import autoencoder
from denoising_autoencoder import denoising_autoencoder
from contractive_autoencoder import contractive_autoencoder

def plot_net_output(net, stats):
    ### Plot loss changes in time.
    plt.subplot(1, 1, 1)
    plt.plot(stats)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    
    ### Sample and visualize some of the learned filter responses.
    W = net.weights['W'].transpose()
    for i in range(0,10):
        for j in range(0,10):
            plt.subplot(10, 10, i*10+j+1)
            rand_index = np.random.random_integers(0,W.shape[0]-1,1)
            plt.imshow(W[rand_index].reshape(28,28), cmap=plt.get_cmap('gray'))
            plt.axis('off')
    plt.show()
 
def __autoencoder_mnist__():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val   = X_val.reshape(X_val.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)    
    
    print 'Train data shape: ',   X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Test data shape: ',    X_test.shape
    print 'Test labels shape: ',  y_test.shape
    print ''
    
    ninput  = 28*28
    nhidden = 300 

    net = autoencoder(layer_units=(ninput, nhidden, ninput), bias=False, act_func = 'sigmoid', 
                      loss_type='cross_entropy', seed=12)
    tic = time.time()
    stats = net.train_with_SGD(X_train, learning_rate=0.2, learning_rate_decay=0.95, reg=0.001, 
                               num_iters=1000, batchsize=128, mu=0.9)
    toc = time.time()
    print toc-tic, 'sec elapsed'
    
    plot_net_output(net, stats)
    
    ### Reconstruct some of the images
    plt_index=1
    for i in range(0,10):
        rand_index = np.random.random_integers(0,X_train.shape[0]-1,1)
        x = X_train[rand_index]
        x_recon = net.predict(x)
        
        plt.subplot(10,2,plt_index)
        plt.imshow(x.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        if i == 0: plt.title('input')
        plt_index+=1
        
        plt.subplot(10,2,plt_index)
        plt.imshow(x_recon.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        if i == 0: plt.title('reconstruction')
        plt_index+=1
    plt.show()
    
    
def __denoising_autoencoder_mnist__():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val   = X_val.reshape(X_val.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)    
    
    print 'Train data shape: ',   X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Test data shape: ',    X_test.shape
    print 'Test labels shape: ',  y_test.shape
    print ''
    
    ninput  = 28*28
    nhidden = 300 

    net = denoising_autoencoder(layer_units=(ninput, nhidden, ninput), bias=True,
                                act_func = 'sigmoid', loss_type='cross_entropy', seed=12)
    tic = time.time()
    stats = net.train_with_SGD_with_noise(X_train, noise=gaussiannoise(rate=0.2, sd=0.2), learning_rate=0.2, 
                                          learning_rate_decay=0.95, reg=0.001, num_iters=1000, batchsize=128, mu=0.9)
    toc = time.time()
    print toc-tic, 'sec elapsed'
    
    plot_net_output(net, stats)
    
    ### Reconstruct some of the images
    plt_index=1
    for i in range(0,10):
        rand_index = np.random.random_integers(0,X_train.shape[0]-1,1)
        x = X_train[rand_index]
        x_recon = net.predict(x)
        
        plt.subplot(10,2,plt_index)
        plt.imshow(x.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        if i == 0: plt.title('input')
        plt_index+=1
        
        plt.subplot(10,2,plt_index)
        plt.imshow(x_recon.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        if i == 0: plt.title('reconstruction')
        plt_index+=1
    plt.show()
    
    
def __contractive_autoencoder_mnist__():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val   = X_val.reshape(X_val.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)    
    
    print 'Train data shape: ',   X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Test data shape: ',    X_test.shape
    print 'Test labels shape: ',  y_test.shape
    print ''
    
    ninput  = 28*28
    nhidden = 300 

    net = contractive_autoencoder(layer_units=(ninput, nhidden, ninput), bias=True, act_func = 'sigmoid', 
                                  loss_type='cross_entropy', seed=12)
    tic = time.time()
    stats = net.train_with_SGD(X_train, learning_rate=0.2, learning_rate_decay=0.95, reg=0.001, 
                               num_iters=1000, batchsize=128, mu=0.9)
    toc = time.time()
    print toc-tic, 'sec elapsed'
    
    plot_net_output(net, stats)
    
    ### Reconstruct some of the images
    plt_index=1
    for i in range(0,10):
        rand_index = np.random.random_integers(0,X_train.shape[0]-1,1)
        x = X_train[rand_index]
        x_recon = net.predict(x)
        
        plt.subplot(10,2,plt_index)
        plt.imshow(x.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        if i == 0: plt.title('input')
        plt_index+=1
        
        plt.subplot(10,2,plt_index)
        plt.imshow(x_recon.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        if i == 0: plt.title('reconstruction')
        plt_index+=1
    plt.show()


# ########################### #
# TEST DIFFERENT AUTOENCODERS #
# ########################### # 

#__autoencoder_mnist__()
#__denoising_autoencoder_mnist__()
__contractive_autoencoder_mnist__()