# AUTOMATIC COLORING OF GREYSCALE IMAGES (WITH THE HELP OF A REFERENCE IMAGE)

Image coloring is a difficult problem. 
Two objects with different colors can appear to be the same in grayscale layout. Coloring of such objects often requires human input to be successful. This project presents a method for colorizing grayscale images by transferring color from a reference image without direct user input. The method colorizes one or more grayscale images, based on a segmented reference color image. 

## Overview of the project
* We represent images in the YUV color space, rather than the RGB color space. In this color space, Y represents luminance and U and V represent chrominance.
* The image is segmented, and each pixel is assigned a label which is its segment. Each segment is roughly uniform in color and texture.
* Discrete Cosine Transform (DCT) coefficients of a pixel's neighbourhood are used as its feature vector. The DCT transformation is applied to the luminance channel of the image.
* For each given unseen feature vector in the test image (the image that needs to be colored), the nearest neighbouring feature vectors from the reference image are found using K Nearest Neighbours. Before the KNN algorithm, the features are brought to lower dimensionality using PCA.
* The labeling of pixels by KNN is further enhanced by spatial smoothing. This means that the labels assigned to the neighbouring pixels in the test image are taken into account. In simpler words, we do something along the lines of if most of the surrounding pixels belong to segment 5, the given pixel should also belong to the same segment.
* Finally, a color is assigned to the pixel. The color value is set as a weighted average of the colors of the pixel's neighbourhood.

A detailed explanation (in IEEE format) can be found in the PDF file 'Automatic Image Coloring'.