# py_autoencoders

This code repository contains "AutoEncoder(AE), Denoising AutoEncoder(DAE), Contractive AutoEncoder (CAE), Contractive Higher-Order(2nd order) AutoEncoder (CAE+H)" written on python. The codes extensively use the lecture notes and base code infrastructures in CS231 Stanford (http://cs231n.stanford.edu/) and CENG 783 METU (http://www.kovan.ceng.metu.edu.tr/~sinan/DL/).

----Some of the features that you can find in the codes----
- "Euclidean" and "Cross Entropy" loss options are selectable in code. 
- Use of bias in computations is selectable.
- Shared weights are used in the course of mapping input to hidden layer and hidden to output layer.
- test.py includes small examples for four autoencoder types run on MNIST dataset.
- You should previously install required packages such as "numpy" from the web.

In case of any failure/recommendation, please don't hesitate to connect with the author (Savas Ozkan / savasozkan.com).

AutoEncoder (loss =179.18)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/ae_filter.png?raw=true "Title")

Denosing AutoEncoder (loss =178.81)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/dae_filter.png)

Contractive AutoEncoder (loss =135.59)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/cae_filter.png)

Contractive Higher-Order AutoEncoder (loss =130.00)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/cae_h_filter.png)
