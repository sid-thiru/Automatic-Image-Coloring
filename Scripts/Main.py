import numpy as np
from skimage import io, color
from Phase_1 import *
from Phase_2 import *
import matplotlib.pyplot as plt

sampling_side = 7
X_points = 100
Y_points = 100
n = 7
max_conf_val = 0.0

train_img_filename = 'E:/Suhas/Main/IUB/4th Sem/ML/Project/train1.jpg'
test_img_filename = 'E:/Suhas/Main/IUB/4th Sem/ML/Project/test1.jpg'

#load the training image
train_img_lab = color.rgb2lab(io.imread(train_img_filename))

#load the test image and convert it to luminance
test_img_lum = color.rgb2lab(io.imread(test_img_filename))[:, :, 0]

#Luminance mapping
train_img_lab[:, :, 0] = luminance(train_img_lab[:, :, 0], test_img_lum)

print('Computing the DCT coefficients of training image..')
dct_vals, labels = GetFeatures(train_img_lab)

print('Building the feature space..')
feature_trans = feature_select(dct_vals, labels[:, 0])
train_vals = feature_trans(dct_vals)

print('Computing the DCT coefficients and feature space of the test image..')
test_img_dct = ImageDct(test_img_lum)
shape = np.shape(test_img_dct)
test_pixels = feature_trans( np.reshape(test_img_dct, (shape[0]*shape[1], -1) ) )

print('Computing the nearest neighbor using KNN..')
feat_spc_labels, smallest_dists, closest_training = knn(test_pixels, train_vals, labels[:, 0])
feat_spc_labels = np.reshape(feat_spc_labels, (shape[0], shape[1]))
smallest_dists = np.reshape(smallest_dists, (shape[0], shape[1]))
closest_training = np.reshape(closest_training, (shape[0], shape[1]))

print('Voting for each test pixel..')
weights = weight_calculate(smallest_dists)

print('Dominant label calculation for each test pixel..')
dom_labels, confidences = voting_of_image(feat_spc_labels, weights)

print('Final coloring of the image..')
colors, confident_pixels = color_optimize(dom_labels, confidences, weights, closest_training, labels[:, 1:])
output = np.zeros((colors.shape[0], colors.shape[1], 3))
n = (sampling_side-1)/2
output[:, :, 0] = test_img_lum[4*n: -4*n, 4*n: -4*n]
output[:, :, 1:] = colors

plt.figure(2)
plt.imshow(feat_spc_labels)
plt.axis('off')
plt.savefig('E:/Suhas/Main/IUB/4th Sem/ML/Project/feature_labels.png')

plt.figure(3)
plt.imshow(dom_labels)
plt.axis('off')
plt.savefig('E:/Suhas/Main/IUB/4th Sem/ML/Project/dominant_labels.png')

plt.figure(4)
plt.imshow(confidences)
plt.axis('off')
plt.savefig('E:/Suhas/Main/IUB/4th Sem/ML/Project/confidence.png')

plt.figure(5)
plt.imshow(color.lab2rgb(output))
plt.axis('off')
plt.savefig('E:/Suhas/Main/IUB/4th Sem/ML/Project/final_output.png')