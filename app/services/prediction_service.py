import numpy as np
import tensorflow as tf
from pathlib import Path
import os

# Load the trained model
# path = "C:\\Users\\udaym\\Documents\\Assignments\\ASE\\SustainableCityManagement_Backend\\app\\saved_models\\saved_model_test.h5"
current_dir = os.path.dirname(os.path.abspath(__file__))  

# Go one directory back (parent directory)
parent_dir = os.path.dirname(current_dir)  

# Construct the path to the model (assuming it's in a subfolder like 'saved_models')
path = os.path.join(parent_dir, "saved_models", "saved_model_test.h5")  
model = tf.keras.models.load_model(
    path,
    compile=True,
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
)

# Function to make predictions
def make_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return float(prediction[0][0])
