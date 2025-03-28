import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Define class labels
class_labels = ['Apple', 'Banana']
TF_ENABLE_ONEDNN_OPTS=0

model_path = r"C:/Users/Admin/ML_flask/app/models/deep_learning/my_model.keras"

if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)



if len(sys.argv) < 3:
    print("Error: Image path and model name required.")
    sys.exit(1)

image_path = sys.argv[1]
model_name = sys.argv[2]
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)
model = tf.keras.models.load_model(model_path)

img = Image.open(image_path)
img = img.resize((150, 150)) 
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0 


prediction = model.predict(img_array)
predicted_class = class_labels[int(prediction[0][0] > 0.5)]

print(predicted_class)
