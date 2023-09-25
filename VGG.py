from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np


def tags(img_path):
    # Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet')

    # Load and preprocess the image
    #img_path = '/Users/anton/Desktop/Personal/AI Projects/Photos/Photos/mist.jpg'  # Replace with the path to your image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get model predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Get only the top predicted class

    # Print the top predicted class and its probability
    top_prediction = decoded_predictions[0]

    print(f"#{top_prediction[1]}")

    return "#" + top_prediction[1]