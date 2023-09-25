import cv2 as cv
import os
import matplotlib.pyplot as plt


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
        return "#vibrant #bright"
    else:
        return "#moody #dark"
    

folder_path = '/Users/anton/Desktop/Personal/AI Projects/Photos/Photos'
image_files_array = get_image_files_in_folder(folder_path)

#for image_path in image_files_array:
#    dominant_colors = getDominantColors(image_path)
#    tag = tag_image_by_color(dominant_colors)
#    print(f"{image_path}     Tag:", tag)

images_with_tags = []

# Iterate through image files and store images with tags
for image_path in image_files_array:
    dominant_colors = getDominantColors(image_path)
    tag = tag_image_by_color(dominant_colors)
    
    img = cv.imread(image_path)  # Read the image using cv2
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv.resize(img, (254, 254))  # Resize the image to 28x28 pixels

    
    images_with_tags.append((img, tag))

