# cnn_cifar10_squential_model

Hello! viewers , welcome to this repository

In this project, we will build a Convolutional Neural Network (CNN) using Keras to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 images of size 32x32 pixels across 10 categories such as airplanes, cars, birds, and cats etc.

Steps to Build the CNN Model:

Load the CIFAR-10 Dataset:
Import the dataset using keras.datasets.cifar10.load_data().
Normalize pixel values to the range [0,1] by dividing by 255.

Define the CNN Model (Sequential API):
Convolutional Layers: Extract spatial features using Conv2D.
Activation Function: Use ReLU to introduce non-linearity.
Pooling Layers: Reduce dimensions using MaxPooling2D.
Dropout Layers: Prevent overfitting by randomly dropping units.

Flatten and Dense Layers: Fully connected layers for classification.
Output Layer: Use softmax activation for 10-class classification.

Compile the Model:
Use the Adam optimizer for efficient learning.
Use categorical_crossentropy as the loss function.
Track accuracy during training.

Train the Model:
Fit the model using model.fit() with training data.
Validate using the test dataset.
