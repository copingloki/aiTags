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

    return (test)

img_path = '/Users/anton/Desktop/Personal/AI Projects/Photos/Photos/sunflower.jpg'
img = mpimg.imread(img_path)

tag_cifar = tags(img_path)

imgplot = plt.imshow(img)
plt.yticks([])
plt.xticks([])
plt.xlabel(tag_cifar)
plt.show()




'''
def getDominantColors(image_path):
    img = cv.imread(image_path)
    scaled_image = cv.resize(img, (100, 100))
    #creates a 1D array of RGB
    pixels = scaled_image.reshape(-1, 3)

    num_clusters = 2
    #below is the kmeans portion, returns dominant colors of the image
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv.kmeans(
        pixels.astype("float32"),
        num_clusters,
        None,
        criteria,
        10,
        cv.KMEANS_RANDOM_CENTERS,
    )

    dominant_colors = centers.astype(int)

    return dominant_colors

def tag_image_by_color(dominant_colors):
    # Define color ranges and corresponding tags
    color_ranges_tags = {
    # Vibrant Colors
    ((200, 0, 0), (255, 150, 150)): "vibrant",  # Red and similar shades
    ((200, 100, 0), (255, 200, 150)): "vibrant",  # Orange and similar shades
    ((200, 0, 100), (255, 200, 255)): "vibrant",  # Pink and similar shades
    ((200, 200, 0), (255, 255, 150)): "vibrant",  # Yellow and similar shades
    ((0, 200, 0), (150, 255, 150)): "vibrant",  # Green and similar shades
    ((0, 200, 200), (150, 255, 255)): "vibrant",  # Cyan and similar shades
    ((200, 0, 50), (255, 150, 200)): "vibrant",  # Magenta and similar shades

    # Moody Colors
    ((0, 0, 0), (50, 50, 50)): "moody",  # Black and similar shades
    ((0, 0, 100), (50, 50, 150)): "moody",  # Dark Blue and similar shades
    ((50, 0, 50), (100, 0, 100)): "moody",  # Deep Purple and similar shades
    ((0, 50, 0), (50, 100, 0)): "moody",  # Dark Green and similar shades
    ((0, 0, 50), (50, 50, 100)): "moody",  # Navy Blue and similar shades
    }
    

    vibrant_count = 0
    moody_count = 0

    for color in dominant_colors:
        assigned_tag = None
        for color_range, tag in color_ranges_tags.items():
            if all(start <= value <= end for start, end, value in zip(color_range[0], color_range[1], color)):
                assigned_tag = tag
                break
        if assigned_tag:
            if assigned_tag == "vibrant":
                vibrant_count += 1
            elif assigned_tag == "moody":
                moody_count += 1

        #print(moody_count)
    if vibrant_count >= moody_count:
        return "vibrant"
    else:
        return "moody"
    '''
'''
# Iterate through image files and store images with tags
for image_path in image_files_array:

    dominant_colors = getDominantColors(image_path)
    tag = tag_image_by_color(dominant_colors)

    img = cv.imread(image_path)  # Read the image using cv2
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv.resize(img, (32, 32))  # Resize the image to 28x28 pixels

    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    test = class_names[index]
    print(f'Prediction is {class_names[index]}')
    
    images_with_tags.append((img, test))
'''


'''
# Create a grid of subplots to display images with tags
num_images = len(images_with_tags)
num_rows = (num_images - 1) // 4 + 1
fig, axes = plt.subplots(num_rows, 4, figsize=(5, 1 * num_rows))

# Iterate through the images and plot them in the grid
for i, (img, test) in enumerate(images_with_tags):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    
    ax.imshow(img)  # Display the image
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(test, fontsize=12, color='white', backgroundcolor='black')
    
# Adjust layout and display the grid
plt.tight_layout()
plt.show()




img = cv.imread('/Users/anton/Desktop/Personal/AI Projects/Photos/Photos/horse.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
scaled_img= cv.resize(img, (32, 32))

plt.imshow(scaled_img, cmap=plt.cm.binary)

prediction = model.predict(np.array([scaled_img]) / 255)
index = np.argmax(prediction)
#print(f'Prediction is {class_names[index]}')
'''