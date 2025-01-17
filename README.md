# Cynaptics Induction Task



## Task List
 - [AI vs Real Image Classifier](#AI-vs-Real)
 - [GAN](#GAN)

## Requirements
Ensure you have the following installed on your system:

 - PyTorch

 - torchvision

 - NumPy

 - Pillow (PIL)

 - CUDA (if using a GPU)

## AI vs Real
This is an image classification program built with PyTorch. The program classifies images into two classes: AI and Real. The training of the convolutional neural network (CNN) on labeled training data is performed, along with generating predictions on test data.

## Program Workflow
1. **Dataset Preparation**
   - Transforming the images by:
     - Resizing them to 200x200.
     - Converting them to tensors.
     - Normalizing to [-1, 1].
   - Loading training data from `data/Train/` using `ImageFolder`.

2. **Neural Network Architecture**
   - Defining the CNN architecture by:
     - 4 convolutional layers.
     - Max-pooling after each convolution.
     - Flatten layer followed by 7 fully connected layers.
     - Output layer for 2 classes ("AI" or "Real").

4. **Training Process**
   - Loss Functionn used is Cross-Entropy Loss.
   - Optimizer used is Adam with learning rate `0.001` and weight decay `1e-5`.
   - For 15 epochs:
     - It performs forward pass to compute predictions.
     - It computes and backpropagates loss.
     - Also, it updates the model weights using the optimizer.
     - Further, it calculates the epoch loss and accuracy.

5. **Saving the Trained Model**
   - Save the model’s state dictionary to `trained_net.pth`.

6. **Loading and Evaluating the Model**
   - We reload the model from the saved state dictionary.
   - Then we switch the model to evaluation mode for inference.

7. **Prediction on Test Data**
   - We load test images from `data/Test/`.
   - Then we preprocess the images and run predictions through the model.
   - Finally, we save the predictions (`Id`, `Label`) to `predictions.csv`..
## Model Architecture

The CNN model architecture is detailed below;

| **Layer Type**     | **Input Dimensions** | **Output Dimensions** | **Kernel Size** | **Activation** |
|--------------------|----------------------|-----------------------|-----------------|----------------|
| Convolutional 1    | 3x200x200           | 12x196x196            | 5x5             | ReLU           |
| Max Pooling        | 12x196x196          | 12x98x98              | 2x2             | -              |
| Convolutional 2    | 12x98x98            | 24x94x94              | 5x5             | ReLU           |
| Max Pooling        | 24x94x94            | 24x47x47              | 2x2             | -              |
| Convolutional 3    | 24x47x47            | 48x43x43              | 5x5             | ReLU           |
| Max Pooling        | 48x43x43            | 48x21x21              | 2x2             | -              |
| Convolutional 4    | 48x21x21            | 96x17x17              | 5x5             | ReLU           |
| Max Pooling        | 96x17x17            | 96x8x8                | 2x2             | -              |
| Flatten            | 96x8x8              | 6144                  | -               | -              |
| Fully Connected 1  | 6144                | 2000                  | -               | ReLU           |
| Fully Connected 2  | 2000                | 1000                  | -               | ReLU           |
| Fully Connected 3  | 1000                | 500                   | -               | ReLU           |
| Fully Connected 4  | 500                 | 256                   | -               | ReLU           |
| Fully Connected 5  | 256                 | 64                    | -               | ReLU           |
| Fully Connected 6  | 64                  | 16                    | -               | ReLU           |
| Output Layer       | 16                  | 2                     | -               | -              |

### Notes:
1. The model uses **ReLU** activation for all intermediate layers except for the output layer.
2. The final output layer has 2 neurons, representing the two classes: "AI" and "Real."
3. Pooling operations reduce the spatial dimensions of the data after each convolutional layer.

This network is designed for classification tasks on 200x200 RGB images.
- **Conv2d**: Applying 2D convolution on input images.
- **MaxPool2d**: Reducing the spatial dimensions using max pooling.
- **Flatten**: Converting the 3D tensor to a 1D vector for fully connected layers.
- **Linear**: Fully connected layers for classification.

The model accepts input images of shape `(3, 200, 200)` and outputs class probabilities for 2 categories: **AI** and **Real**.

## Acknowledgements

[NeuralNine Video on Image Classification on CIFAR10 dataset](https://www.youtube.com/watch?v=CtzfbUwrYGI&t=667ss)



## GAN
## Overview
The GAN consists of:
- **Generator**: This maps random noise (`z_dim`) to a 64x64 RGB image using a series of fully connected layers with ReLU activations.
- **Discriminator**: This classifies images as real or fake, using fully connected layers with LeakyReLU activations and a final Sigmoid layer.
## Program Workflow.

## Data Loading

- The dataset is loaded using `torchvision.datasets.ImageFolder`.
- The images are resized to 64x64, normalized to [-1, 1] and then converted to tensors.

## Model Initialization

Two neural networks are initialized:
- **Generator**: This maps random noise vectors to 64x64 images.
- **Discriminator**: This predicts whether an image is real or fake.

## Training

- **Discriminator**: The discriminator is trained to distinguish real images from fake ones.
- **Generator**: The generator, on the other hand is trained to produce images that can fool the Discriminator.
- Losses (`lossD` and `lossG`) are calculated using Binary Cross-Entropy Loss.

## Image Saving

- A fixed noise vector is used to generate images during training.
- Generated images are saved in the `generatedimages/` directory every 50 batches.
## Model Architecture.
### Discriminator

| Layer              | Input Size       | Output Size      | Activation    |
|--------------------|------------------|------------------|---------------|
| Fully Connected    | 3 x 64 x 64      | 512              | LeakyReLU(0.2)|
| Fully Connected    | 512              | 256              | LeakyReLU(0.2)|
| Fully Connected    | 256              | 1                | Sigmoid       |

### Generator

| Layer              | Input Size | Output Size | Activation    |
|--------------------|------------|-------------|---------------|
| Fully Connected    | z_dim      | 128         | ReLU          |
| Fully Connected    | 128        | 256         | ReLU          |
| Fully Connected    | 256        | 512         | ReLU          |
| Fully Connected    | 512        | 3 x 64 x 64 | Tanh          |

### Hyperparameters

| Parameter          | Value        |
|--------------------|--------------|
| Learning Rate      | 1e-4         |
| Latent Vector Dim. | 100          |
| Image Dim.         | 3 x 64 x 64  |
| Batch Size         | 32           |
| Epochs             | 200          |


## Acknowledgements





[Aladdin Pearson Video on GANs](https://www.youtube.com/watch?v=OljTVUVzPpM&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=2)
