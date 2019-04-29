# PROGRAMMER: Mertens Moreno, Carlos Edgar
# DATE CREATED: 04/24/2019
# REVISED DATE:
# PURPOSE: Predict an image of a cell blood if it is infected with malaria
#			parasite or not.
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --input <path to a cell blood image> --checkpoint 
# 							<path to checkpoint file loading pre-trained model>

# Imports
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import argparse


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

    parser.add_argument('--checkpoint', type=str, default='Saved_models/malaria_cnn.h5',
                        help='directory to load checkpoints')
    parser.add_argument('--input', type=str,
                        default='Data/Input-Data/Healthy/healthy4.png', 
						help='path to an image')

    return parser.parse_args()


# Call function to get command line arguments
in_arg = get_args()


def path_to_tensor(img_path):
    """Convert an image into 4D tensor."""
    
    # Load image path input into a (224, 224) image
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert the image into a 3D tensor image (224, 224, 3)
    x = image.img_to_array(img)
    
    # Convert 3D tensor to 4D tensor (1, 224, 224, 3) and return
    return np.expand_dims(x, axis=0)


# Load neural network model saved
model = load_model(in_arg.checkpoint)

# Get path to input image to be predicted 
input_image = in_arg.input

# Calculate prediction for image input
image = path_to_tensor(input_image)
y_pred = model.predict(image)

# Print the infected percentage prediction
infected_perc = round(y_pred.tolist()[0][1] * 100, 2)
print('\nThe image is {}% predicted to be infected'.format(infected_perc))
