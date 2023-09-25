import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models
import os
import matplotlib.image as mpimg

# Load CIFAR-10 data
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar100.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

#class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

class_names = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud",
    "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree",
    "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter",
    "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm"]


model = models.load_model('image_classifier.model')

def is_image_file(filename):
    # Define the list of image file extensions you want to include
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    # Check if the filename's extension matches any image extension
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def get_image_files_in_folder(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and is_image_file(filename):
            image_files.append(file_path)
    return image_files

folder_path = '/Users/anton/Desktop/Personal/AI Projects/Photos/Photos'
image_files_array = get_image_files_in_folder(folder_path)

images_with_tags = []

def tags(image_path):

    img = cv.imread(image_path)  # Read the image using cv2
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv.resize(img, (32, 32))  # Resize the image to 32x32 pixels

    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    test = class_names[index]
    #print(f'Prediction is {class_names[index]}')

    return "#" + (test)

#img_path = '/Users/anton/Desktop/Personal/AI Projects/Photos/Photos/sunflower.jpg'
#img = mpimg.imread(img_path)

#tag_cifar = tags(img_path)

#imgplot = plt.imshow(img)
#plt.yticks([])
#plt.xticks([])
#plt.xlabel(tag_cifar)
#plt.show()