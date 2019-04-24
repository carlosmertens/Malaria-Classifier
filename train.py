# PROGRAMMER: Mertens Moreno, Carlos Edgar
# DATE CREATED: 04/24/2019
# REVISED DATE:
# PURPOSE: Train a Neural Network Image Classifier with Keras to predict an image
# 			classifier. It will be trained with a dataset images of blood cell
# 			infected and not infected with malaria parasite.
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --save_dir <directory to save the checkpoint> --dataset_dir
# 		<path to the dataset images directory> --epochs <epochs to train>
#   Example call:
#    >>python train.py --save_dir Saved_models --dataset_dir Data/ --epochs 20

# Import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Import modules for data processing
import os
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Import modules to build neural network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

# Import modules for evaluation metrics
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

# Import module to save the trained model
from keras.models import load_model


def get_args():
    """
    Function to retrieve and parse the command line arguments,
    then to return these arguments as an ArgumentParser object.
    Parameters:
     None.
    Returns:
     parser.parse_args(): input or default argument objects.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str,
                        default='Data/', help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='Saved_models/',
                        help='directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=20,
                        help='hyperparameters for epochs')

    return parser.parse_args()


# Call function to get command line arguments
in_arg = get_args()


def prep_data(uninfected, parasitized):
    """
    Function to prepare the data for training and testing. Use transforms to normalized
    the data and ImageFolder to retrieve the data.
    Parameters:
     train_path, valid_path and test_path: data sets to be prepare.
    Returns:
     train_loader, valid_loader, test_loader: Sets ready to be use by the model.
     train_datasets: to be used to classify the labels.
    """

    # Load images with glob module
    healthy_cell = glob(uninfected, recursive=True)
    infected_cell = glob(parasitized, recursive=True)

    print(len(healthy_cell), "Healthy cell images loaded")
    print(len(infected_cell), "Infected cell images loaded\n")

    # Create empty list to hold images processed
    features = []
    labels = []

    # Loop into the images and resize, convert them into matrix (ndarray type)
    for img in healthy_cell:
        image = cv2.imread(img)
        image_resized = cv2.resize(
            image, (224, 224), interpolation=cv2.INTER_CUBIC)
        features.append(image_resized)
        labels.append(0)

    for img in infected_cell:
        image = cv2.imread(img)
        image_resized = cv2.resize(
            image, (224, 224), interpolation=cv2.INTER_CUBIC)
        features.append(image_resized)
        labels.append(1)

    # Convert images matrix into array
    features = np.array(features)
    labels = np.array(labels)

    return features, labels


# Define dataset directories' path
data_dir = in_arg.dataset_dir
uninfected_path = data_dir + 'Uninfected/*.png'
parasitized_path = data_dir + 'Parasitized/*.png'

# Call function to load and prepare the data
features, labels = prep_data(uninfected_path, parasitized_path)

# Split data into training and testing with train_test_split module
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# Convert testing sets into binary category
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define model's architecture
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())

model.add(Dense(500, activation='relu'))

model.add(Dense(2,activation='softmax'))

# Compile optimizer, loss and metrics
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Print model created
print("\n***** Neural Network Model *****\n")
model.summary()

# Print message to start training the model
print("\n*** Model will start training ***\n")

# Specify the number of epochs that you would like to use to train the model.
epochs = in_arg.epochs

# Save model checkpoints
checkpointer = ModelCheckpoint(filepath=in_arg.save_dir + 'weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)

# Train neural network
model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=32, callbacks=[checkpointer], verbose=1)

# Calculate test accuracy with the test datasets
accuracy = model.evaluate(X_test, y_test, verbose=1)
print("\nTest Accuracy:", accuracy[1])

# Calculate prediction on the test dataset
y_pred = model.predict(X_test)

# Print classification report with Scikit-learn library
print("\n*** Metric Evaluation Report ***")
print(classification_report(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1)))

# Calculate F-beta score with Scikit-learn library
fbeta = fbeta_score(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis=1), beta=0.92)
print("F-beta score:", fbeta)

# Save model into the directory assigned
model.save(in_arg.save_dir + 'malaria_cnn.h5')
