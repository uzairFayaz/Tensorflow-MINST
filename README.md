# MNIST Digit Classification using TensorFlow

This project trains a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is grayscale and has a size of 28x28 pixels.

## Project Overview
1. **Load the Dataset**: The dataset is loaded using TensorFlow's built-in MNIST dataset.
2. **Preprocess the Data**:
   - Normalize pixel values to the range [0,1] for better training.
   - Reshape images to add a channel dimension (28x28x1) since CNNs expect multi-dimensional inputs.
3. **Define the CNN Model**:
   - Three convolutional layers with ReLU activation.
   - MaxPooling layers to reduce spatial dimensions.
   - Flattening layer to convert the feature map into a 1D array.
   - Fully connected dense layers for classification.
   - Softmax activation for output probabilities.
4. **Compile and Train**:
   - Uses Adam optimizer and sparse categorical cross-entropy loss.
   - Trains for 5 epochs with validation.
5. **Evaluate the Model**:
   - Measures accuracy on the test dataset.
   - Displays accuracy progression using a plot.

## Requirements
- Python
- TensorFlow
- Matplotlib

## Running the Code
Simply execute the Python script to train the model:
```bash
python mnist_model.py
```

## Output
- The trained model will classify handwritten digits with high accuracy.
- A plot showing training and validation accuracy per epoch.

## License
This project is open-source and free to use for learning purposes.

