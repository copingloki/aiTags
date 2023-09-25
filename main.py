from cv2 import imshow
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.image as mpimg
import VGG
import cifar
import colors

img_path = '/Users/anton/Desktop/Personal/AI Projects/Photos/Photos/sunset.jpg'
img = mpimg.imread(img_path)

dominant_colors = colors.getDominantColors(img_path)

tag_vgg_initial = VGG.tags(img_path)
tag_vgg_mod = tag_vgg_initial.replace("_", "")
tag_vibe = colors.tag_image_by_color(dominant_colors)
tag_cifar = cifar.tags(img_path)

imgplot = plt.imshow(img)
plt.yticks([])
plt.xticks([])
plt.xlabel(tag_vibe + " " + tag_vgg_mod + " " + tag_cifar + " #photography #art #photooftheday")
plt.show()
