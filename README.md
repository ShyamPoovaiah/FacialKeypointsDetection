[//]: # (Image References)
[single layer loss image]: ./results/losses/single_hidden_layer.png "Single Layer Loss"
[single layer model image]: ./results/models/single_hidden_layer.png "Single Layer Model"
[single layer accuracy image]: ./results/accuracy/single_hidden_layer.png "Single Layer Model"
[single layer prediction image]: ./results/predictions/single_hidden_layer.png "Single Layer Model"
[convolution loss image]: ./results/losses/convolution.png "Convolution Loss"
[convolution model image]: ./results/models/convolution_layer.png "Convolution Model"
[convolution accuracy image]: ./results/accuracy/convolution.png "Convolution Accuracy"
[optimized convolution loss image]: ./results/losses/convolution_optimized.png "Optimized Convolution Loss"
[optimized convolution model image]: ./results/models/optimized_convolution_layer.png "Optimized Convolution Model"
[optimized convolution accuracy image]: ./results/accuracy/optimized_convolution.png "Optimized Convolution Accuracy"
[flipped]: ./results/other/flip_without_adjust.png "Flipped"
# Facial Keypoints Detection

## Motive
Faces have always intringued me. From reading Paul Ekman's 'Unmasking the face' to staring at my opponent's face across the ring unable to decide if he is really going to throw that punch, it is always exciting. After joining a 'Computational Machine Learning' course at 
'Indian Institute of Science' we decided to go work on faces.

## Contents
Prerequisites   
Data : Loading and Normalization.   
First model : A Dense network with single hidden layer.   
Testing a generated Model.   
Second model : Using CNN's   
Data Augmentation to improve training accuracy.   
Changing learning rate and momentum over time   
Using Dropout while training.   
Third Model : Ensemble Learning with specialists for each feature.   
Supervised pre-training   
Conclusion   

## Prerequisites
1. Download the files from https://www.kaggle.com/c/facial-keypoints-detection/data. Copy the files to the 'data folder' before running any of the training code.   
2. Install python, numpy, matplotlib, pandas, keras and tensorflow.
3. Install 'pydot' to visualize models from keras.
```
pip3 install pydot
```
4. 'pydot' requires a system level installation of 'GraphViz'.
```
brew install Graphviz
```

## Data

The training dataset consists of 7,049 96x96 gray-scale images. The model is supposed learn to find the correct position (the x and y coordinates) of 15 keypoints, such as left_eye_center, right_eye_outer_corner, mouth_center_bottom_lip etc....
For some of the keypoints we only have about 2,000 labels, while other keypoints have more than 7,000 labels available for training.

## Loading the data.
The column containing the image pixels is reshaped into a 2D matrix. 1 refers to number of channels. Since this is a grey scale image its 1.

```python
  X = X.reshape(-1, 96, 96, 1)
```

## First Model : A Single Hidden Layer
A model with the below configuration was used:   
![Single Layer Model][single layer model image]

While training the output looks like this:   
Epoch 395/400
1712/1712 [==============================] - 1s 416us/step - loss: 7.1303e-04 - get_categorical_accuracy_keras: 0.8037 - val_loss: 0.0030 - val_get_categorical_accuracy_keras: 0.7079
Epoch 396/400
1712/1712 [==============================] - 1s 432us/step - loss: 7.1268e-04 - get_categorical_accuracy_keras: 0.8072 - val_loss: 0.0030 - val_get_categorical_accuracy_keras: 0.7079
Epoch 397/400
1712/1712 [==============================] - 1s 433us/step - loss: 7.0165e-04 - get_categorical_accuracy_keras: 0.8096 - val_loss: 0.0031 - val_get_categorical_accuracy_keras: 0.7056
Epoch 398/400
1712/1712 [==============================] - 1s 429us/step - loss: 7.3057e-04 - get_categorical_accuracy_keras: 0.8078 - val_loss: 0.0030 - val_get_categorical_accuracy_keras: 0.7079
Epoch 399/400
1712/1712 [==============================] - 1s 427us/step - loss: 7.0383e-04 - get_categorical_accuracy_keras: 0.8160 - val_loss: 0.0030 - val_get_categorical_accuracy_keras: 0.7220
Epoch 400/400
1712/1712 [==============================] - 1s 436us/step - loss: 7.1510e-04 - get_categorical_accuracy_keras: 0.8078 - val_loss: 0.0030 - val_get_categorical_accuracy_keras: 0.7196
2140/2140 [==============================] - 0s 136us/step

### Notes during training
1. Decision on number of layers and neurons : This [dessertation](https://www.heatonresearch.com/2017/06/01/hidden-layers.html) by Jeff Heaton  was used as a recommendation for the number of neurons and layers.

2. Initial attempt with an Adam optimizer and a learning rate of .1 resulted in an exploding gradients problem. The training converged after reducing the learnign rate to .01 and using 'clipnorm' to control gradient clipping.

3. The 'accuracy' metric throws an error regarding dimension. Providing an implementation to circumvent the same.


The logs of the training are [here](./results/losses/single_hidden_layer.csv)

### Results
![Single Layer Loss][single layer loss image]   
There is a small amount of overfitting, but it is not that bad. In particular, we don't see a point where the validation error gets worse again, thus 'Early stopping', would not be useful. Regularization was not used to control overfitting either.

Accuracy of the model:
![Single Layer Model][single layer accuracy image]   

 Based on MSE loss(Test) of .0029712174083410857, we'll take the square root and multiply by 48 again (since we had normalized locations from [-1, 1])
```python
>>> import numpy as np
>>> np.sqrt(.0029712174083410857)*48
2.61642597999979
```

### Predicting with the model

While loading the model a custom object with the custom accuracy function needs to be passed. Otherwise, a 'Undefined metric function' error is thrown.
```python
model = load_model(MODEL_PATH, custom_objects={'get_categorical_accuracy_keras': get_categorical_accuracy_keras})
```

The prediction yields decent results as seen below:   
![Single Layer Model][single layer prediction image]  

## Second Model : A Convolutinal Neural Network
A model with the below configuration was used:   
![Convolution Model][convolution model image]

### Notes during training
1. The input dimensions would match after setting 'data_format' to 'channels_last'. Also a flattening layer layer was required before the dense layers to keep the dimensions matched.

2. The training on a machine without a GPU took hours. The accuracy and gain did not improve by much after a few hundred epochs. Early stopping could be used.

The logs of the training are [here](./results/losses/convolution_layer.csv)

### Results
![Convolution Loss][convolution loss image]   
There is a small amount of overfitting, but it is not that bad. In particular, we don't see a point where the validation error gets worse again, thus 'Early stopping', would not be useful. Regularization was not used to control overfitting either.

```python
>>> import numpy as np
>>> np.sqrt(.0029712174083410857)*48
2.61642597999979
```

Accuracy of the model:
![Convolution Accuracy][convolution accuracy image]   

### Training with Dropout
![Optimized Convolution Model][optimized convolution model image]
![Optimized Convolution Loss][optimized convolution loss image]  
![Optimized Convolution Accuracy][optimized convolution accuracy image]   
 

## Augmenting the data.
The loss of the model above is not low compared to the Dense Model. This can be attributed to a lot of rows with NA values being dropped. One way to get around this is to augment available data.  
Here the images are flipped to increase the number of training data.      
![Flipped][flipped]   
We can see that the target values also needs to be updated along with flipping the image.

## Training Specialists (TODO)
Instead of training a single model, let's train a few specialists, with each one predicting a different set of target values. We'll train one model that only predicts left_eye_center and right_eye_center, one only for nose_tip and so on; overall, we'll have six models. This will allow us to use the full training dataset.

The six specialists are all going to use exactly the same network architecture. We will use 'Early stopping' since this training is going to take a while. Early stopping will be based on improvement(No improvement rather) in error. Additionally we will  increase the momentum.   
Both these involve callbacks during training.
```python
history = model.fit(X, y, epochs=epochs, verbose=1, batch_size=batch_size, validation_split=.2, shuffle=True, callbacks=[csv_logger, checkpointer, update_momentum, early_stopping])
```

