# Text_Classificaiton
This repository contains the necessary code files for implementing a text classification task. The purpose of this project is to demonstrate the process of text classification, including preprocessing the data, training a neural network model, and using the trained model for classification. The following sections provide an overview of each code file:

**File Descriptions

get_line.py: This script is responsible for reading a text file and extracting each line as a separate data point. It provides a function get_lines(filename) that takes a file path as input and returns a list of lines from the file.

main.py: This is the main script that orchestrates the entire text classification workflow. It imports the necessary modules and defines the high-level logic for data preprocessing, model training, and classification. It utilizes the functions from preprocess.py, nnmodel.py, and training.py to perform these tasks.

nnmodel.py: This module contains the implementation of the neural network model used for text classification. It defines a class NNModel that encapsulates the architecture and functionality of the neural network. The class includes methods for model initialization, forward propagation, and training.

preprocess.py: This module handles the preprocessing of text data before training or classification. It contains functions for tasks such as tokenization, removing stop words, and vectorizing the text data. These preprocessing steps are crucial for converting raw text into a format suitable for training a machine learning model.

training.py: This module provides the necessary functions to train the neural network model on the preprocessed data. It includes functions for splitting the data into training and validation sets, defining the loss function and optimizer, and executing the training loop. The trained model is saved for later use in classification.

** execution.ipynb shows the execution of codes

