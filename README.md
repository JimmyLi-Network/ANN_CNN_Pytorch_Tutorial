# Introduction
This tutorial is a PyTorch implementation of two neural network architectures—an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for fresh students—in a Jupyter Notebook environment. The notebook covers all the essential steps, including data loading and preprocessing, model definition, training loops, and evaluation on a test set. The project uses the MNIST dataset for digit classification, which is ideal for experimenting with both simple and convolution-based models.

## ANN Architecture

- **Input Layer:**  
  The ANN model takes flattened MNIST images as input. Each 28×28 image is reshaped into a 784-dimensional vector.

- **Hidden Layers:**  
  The network includes one or more fully connected (linear) layers. For example, a typical configuration might include:
  - A hidden layer with 128 neurons,
  - Followed by a ReLU activation to introduce nonlinearity.

- **Output Layer:**  
  The final layer is a fully connected layer that maps the hidden representation to 10 output neurons, corresponding to the 10 digit classes. The cross-entropy loss function (which internally applies a softmax) is used during training.

- **Training Details:**  
  The model is trained using the Adam optimizer and cross-entropy loss. The notebook monitors training progress and evaluates performance on the test set.

## CNN Architecture

- **Convolutional Layers:**  
  The CNN model leverages convolutional layers to process the MNIST images:
  - **First Convolutional Block:**  
    - Applies a 3×3 convolution with 32 filters,
    - Followed by Batch Normalization and a ReLU activation,
    - Then a Max Pooling layer (2×2) for downsampling.
  
  - **Second Convolutional Block:**  
    - Applies another 3×3 convolution with 64 filters,
    - Followed by Batch Normalization and a ReLU activation,
    - Then another 2×2 Max Pooling layer.

- **Fully Connected Layers:**  
  After the convolutional blocks, the feature maps are flattened into a vector:
  - A fully connected layer maps the flattened features to a hidden representation (e.g., 128 neurons),
  - Another ReLU activation is applied,
  - The final fully connected layer maps the representation to 10 output classes.

- **Training Details:**  
  Similar to the ANN, the CNN uses cross-entropy loss and the Adam optimizer. The notebook shows how to build, train, and evaluate the CNN model.

## Jupyter Notebook Implementation

- **Data Loading:**  
  The MNIST dataset is downloaded via `torchvision.datasets.MNIST`, with appropriate transforms (conversion to Tensor and normalization) applied. Data is loaded using DataLoader objects to facilitate efficient batching.

- **Model Definition:**  
  Two separate model classes are defined:
  - `SimpleANN` for the ANN,
  - `SimpleCNN` for the CNN.  
  Each class includes a `forward` method that outlines the data flow through the network.

- **Training Loop:**  
  For both models, the training loop:
  - Iterates over epochs and batches,
  - Performs forward propagation, loss calculation, backpropagation, and optimizer updates,
  - Prints running loss to track training progress.
