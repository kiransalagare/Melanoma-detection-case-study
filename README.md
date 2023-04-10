# Melanoma-detection-case-study

The cardinal objective of this project is to develop state of the art Convolutional Neural Network (CNN) model to perform the classiﬁcation of skin lesion images into respective cancer types. The model is trained and tested on the dataset made available by International Skin Imaging Collaboration (ISIC). The model can beused for analyzing the lesion image and ﬁnd out if it’s dangerous at early stage.

Problem Statement
In the skin biopsy, the dermatologist takes some part of the skin lesion and examines it under the microscope. The current process takes almost a week or more, starting from getting a dermatologist appointment to getting a biopsy report. The aims to shorten the current gap to just a couple of days by providing the predictive model. The approach uses Convolutional Neural Network (CNN) to classify nine types of skin cancer from outlier lesions images. This reduction of a gap has the opportunity to impact millions of people positively.

## CNN Architecture Design
To classify skin cancer using skin lesions images. To achieve higher accuracy and results on the classification task, I have built custom CNN model.

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
 conv2d_1 (Conv2D)           (None, 180, 180, 32)      25632     
max_pooling2d (MaxPooling2D) (None, 90, 90, 32)       0                                                                   
conv2d_2 (Conv2D)           (None, 90, 90, 32)        25632     
max_pooling2d_1(MaxPooling 2D) (None, 45, 45, 32)       0                                                         
conv2d_3 (Conv2D)           (None, 45, 45, 64)        51264     
max_pooling2d_2 (MaxPooling 2D)   (None, 22, 22, 64)       0                                                                     
conv2d_4 (Conv2D)           (None, 22, 22, 64)        102464                                                                    
max_pooling2d_3 (MaxPooling 2D)    (None, 11, 11, 64)       0         
dropout (Dropout)           (None, 11, 11, 64)        0                                                       
flatten (Flatten)           (None, 7744)              0                                                                      
dense (Dense)               (None, 9)                 69705         
                                                                                                                               
=================================================================
Total params: 277,129
Trainable params: 277,129
Non-trainable params: 0

## Conclusions
- Conclusion 1 from the analysis
- Conclusion 2 from the analysis
- Conclusion 3 from the analysis
- Conclusion 4 from the analysis

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
-tensorflow- version 2.12.0
- matplotlib.pyplot 
- numpy 
- tensorflow.keras
- pathlib



References
Melanoma Skin Cancer from https://www.cancer.org/cancer/melanoma-skin-cancer/about/what-is-melanoma.html

Introduction to CNN from https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/

Image classification using CNN from https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/

Efficient way to build CNN architecture from https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7























