# Melanoma-detection-case-study

The cardinal objective of this project is to develop state of the art Convolutional Neural Network (CNN) model to perform the classiﬁcation of skin lesion images into respective cancer types. The model is trained and tested on the dataset made available by International Skin Imaging Collaboration (ISIC). The model can beused for analyzing the lesion image and ﬁnd out if it’s dangerous at early stage.

## Table of contents
* Problem Statement
* CNN Architecture Design
* Conclusions
* Technologies Used
* Contact

## Problem Statement
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## CNN Architecture Design
To classify skin cancer using skin lesions images. 

![image](https://user-images.githubusercontent.com/121044079/231839702-406d6dd8-8e57-4cbb-ace9-346845a9c15c.png)

In order to achieve higher accuracy and results on the classification task, a custom CNN model has been built.

Rescalling Layer - To rescale an input in the [0, 255] range to be in the [0, 1] range.
Convolutional Layer - Convolutional layers apply a convolution operation to the input, passing the result to the next layer. A convolution converts all the pixels in its receptive field into a single value. For example, if you would apply a convolution to an image, you will be decreasing the image size as well as bringing all the information in the field together into a single pixel.
Dropout Layer - The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
Flatten Layer - Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector. And it is connected to the final classification model, which is called a fully-connected layer.
Dense Layer - The dense layer is a neural network layer that is connected deeply, which means each neuron in the dense layer receives input from all neurons of its previous layer.
Activation Function(ReLU) - The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better.
Activation Function(Softmax) - The softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability distribution. The main advantage of using Softmax is the output probabilities range. The range will 0 to 1, and the sum of all the probabilities will be equal to one

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 180, 180, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 180, 180, 32)      2432      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 90, 90, 32)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 90, 90, 32)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 90, 90, 64)        18496     
                                                                 
 activation (Activation)     (None, 90, 90, 64)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 88, 88, 64)        36928     
                                                                 
 activation_1 (Activation)   (None, 88, 88, 64)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 44, 44, 64)       0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 44, 44, 64)        0         
                                                                 
 flatten (Flatten)           (None, 123904)            0         
                                                                 
 dense (Dense)               (None, 512)               63439360  
                                                                 
 activation_2 (Activation)   (None, 512)               0         
                                                                 
 dropout_2 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 9)                 4617      
                                                                 
 activation_3 (Activation)   (None, 9)                 0         
                                                                 
=================================================================
Total params: 63,501,833
Trainable params: 63,501,833
Non-trainable params: 0
_________________________________________________________________

## Conclusions
![image](https://user-images.githubusercontent.com/121044079/231836576-f574b3af-669c-4bb2-aa31-4124d5459374.png) 

Conclusion 1: The model is overfitting because the is a visible difference between loss functions in training and test just before the 10th epoch. The accuracy is greater than 60% as there are sufficient features to remember the pattern.

![image](https://user-images.githubusercontent.com/121044079/231836642-43a4aedb-eed1-42af-b32c-a77a76687397.png) 

Data augmentation helped solve the overfitting problem. However, accuracy decreased to 50%.

![image](https://user-images.githubusercontent.com/121044079/231836732-4abd6f9e-2ce9-42bd-bab8-a590345c1638.png)

Using the Augmentor library helped in increasing the accuracy of the training set upto more than 80% but overfitting still persists on the model.

## Technologies Used
tensorflow - version 2.12.0
matplotlib - version 3.7.1 
numpy - version 1.22.4
tensorflow - version 2.12.0
pathlib - version 1.0.1

## Contact
Created by @kiransalagare and @nmadhavashyam - feel free to contact us!























