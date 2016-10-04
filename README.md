# py_autoencoders

This code repository contains "AutoEncoder(AE), Denoising AutoEncoder(DAE), Contractive AutoEncoder (CAE), Contractive Higher-Order(2nd order) AutoEncoder (CAE+H)" written on python. The codes extensively use the lecture notes and base code infrastructures in CS231 Stanford (http://cs231n.stanford.edu/) and CENG 783 METU (http://www.kovan.ceng.metu.edu.tr/~sinan/DL/).

----Some of the features that you can find in the codes----
- "Euclidean" and "Cross Entropy" loss options are selectable in code. 
- Use of bias in computations is selectable.
- Shared weights are used in the course of mapping input to hidden layer and hidden to output layer.
- test.py includes small examples for four autoencoder types run on MNIST dataset.
- You should previously install required packages such as "numpy" from the web.

In case of any failure/recommendation, please don't hesitate to connect with the author (Savas Ozkan / savasozkan.com).

The example results with 2000 iterations 0.1 learning step parameter configurations.
AutoEncoder (euclidean loss=6.6937)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/ae.png)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/ae_re.png)

Denosing AutoEncoder (euclidean loss=6.8654)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/dae.png)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/dae_re.png)

Contractive AutoEncoder (euclidean loss=6.0982)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/cae.png)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/cae_re.png)

Contractive Higher-Order AutoEncoder (euclidean loss=6.0628)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/caeh.png)
![alt tag](https://github.com/savasozkan/py_autoencoders/blob/master/results/caeh_re.png)
