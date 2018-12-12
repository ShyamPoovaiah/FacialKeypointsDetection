[//]: # (Image References)
[single layer loss image]: ./results/losses/single_hidden_layer.png "Single Layer Loss"
[single layer model image]: ./results/models/single_hidden_layer.png "Single Layer Model"
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

## Data

The training dataset consists of 7,049 96x96 gray-scale images. The model is supposed learn to find the correct position (the x and y coordinates) of 15 keypoints, such as left_eye_center, right_eye_outer_corner, mouth_center_bottom_lip etc....
For some of the keypoints we only have about 2,000 labels, while other keypoints have more than 7,000 labels available for training.

## Loading the data.



## First Model : A Single Hidden Layer
A model with the below configuration was used:
![Single Layer Model][single layer model image]

### Notes during training
1. Decision on number of layers and neurons : This [dessertation](https://www.heatonresearch.com/2017/06/01/hidden-layers.html) by Jeff Heaton  was used as a recommendation for the number of neurons and layers.
2. Initial attempt with an Adam optimizer and a learning rate of .1 resulted in an exploding gradients problem. The training converged after reducing the learnign rate to .01 and using 'clipnorm' to control gradient clipping.

The logs of the training are [here](./results/losses/single_hidden_layer.csv)

### Results
![Single Layer Loss][single layer loss image]
There is a small amount of overfitting, but it is not that bad. In particular, we don't see a point where the validation error gets worse again, thus 'Early stopping', would not be useful. Regularization was not used to control overfitting either.

 Based on MSE loss of x, we'll take the square root and multiply by 48 again (since we had normalized locations from [-1, 1])
```python
>>> import numpy as np
>>> np.sqrt(x) * 48
y
```
