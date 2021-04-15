# mnist-raw-classifier
Machine learning classifier that classifies MNIST data using either a Neural Network (NN) or Convolutional Neural Network (CNN) model. Both models are implemented using Keras/Tensorflow and Scikit-learn packages.

## Run
```
python3 classifier.py [mode]
```
Requires Tensorflow, Keras, and Scikit-learn packages installed appropriately for your setup.

## Models
Both models are compiled using a Mean-Squared Error loss function and Keras's "Adam" optimizer.
Both are trained and tested using k-fold cross validation, with 5 folds over 200 epochs.

### NN
Keras Sequential model with:
1. Input layer with 784 input nodes
2. Densely-connected hidden layer with 128 nodes, sigmoid activation
3. Densely-connected output layer 10 nodes, softmax activation

Example learning curves and results:

![NN Accuracy](https://raw.githubusercontent.com/aczolan/mnist-raw-classifier/master/images/NN_Accuracy.png)
![NN Loss](https://raw.githubusercontent.com/aczolan/mnist-raw-classifier/master/images/NN_Loss.png)
```
Fold 1 Accuracy: 86.00000143051147
Fold 2 Accuracy: 88.49999904632568
Fold 3 Accuracy: 88.49999904632568
Fold 4 Accuracy: 87.00000047683716
Fold 5 Accuracy: 87.93969750404358
Mean accuracy: 87.58793950080872
```

### CNN
Keras Sequential model with:
1. Conv2D input layer, with (28, 28, 1) input shape, (4, 4) kernel size, sigmoid activation
2. Pooling layer, pooling size (2, 2)
3. Conv2D layer, (4, 4) kernel size, sigmoid activation
4. Pooling layer, pooling size (2, 2)
5. Flattening layer
6. Densely-connected hidden layer with 32 nodes, sigmoid activation
7. Densely-connectes output layer with 10 nodes, softmax activation

Example learning curves and results:
![CNN Accuracy](https://raw.githubusercontent.com/aczolan/mnist-raw-classifier/master/images/CNN_Accuracy.png)
![CNN Loss](https://raw.githubusercontent.com/aczolan/mnist-raw-classifier/master/images/CNN_Loss.png)
```
Fold 1 Accuracy: 95.49999833106995
Fold 2 Accuracy: 94.49999928474426
Fold 3 Accuracy: 33.000001311302185
Fold 4 Accuracy: 89.49999809265137
Fold 5 Accuracy: 94.47236061096191
Mean accuracy: 81.39447152614594
```