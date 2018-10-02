import numpy as np
from random import randint
from scipy.fftpack import dct
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
sampling_side = 7
X_points = 100
Y_points = 100

def luminance(train, test):
	mean_tr = np.mean(train)
	mean_te = np.mean(test)
	std_tr = np.std(train)
	std_te = np.std(test)
	return std_te * (train - mean_tr) / std_tr + mean_te

def neighbor(image, x, y, size = sampling_side):
	n = (size-1)/2
	x_min = x - n
	x_max = x + n + 1
	y_min = y - n
	y_max = y + n + 1
	neighbors = image[x_min : x_max, y_min : y_max]
	return neighbors

def ImageDct(lum_image):
    n = (sampling_side - 1) / 2
    X = len(lum_image)
    Y = len(lum_image[0])

    features = []
    for x in range(n, X - n):
        row = []
        for y in range(n, Y - n):
            neighbors = neighbor(lum_image,x,y)
            DCT = dct(dct(neighbors.T, norm='ortho').T, norm='ortho')
            feature = np.reshape(DCT, -1).tolist()
            row.append(feature)
        features.append(row)
    return features

def GetFeatures(image_Lab):
    image = np.array(image_Lab)
    flat_image = np.reshape(image, [-1, 3])
    bandwidth = estimate_bandwidth(flat_image, quantile=.2, n_samples=5000)
    ms = MeanShift(bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    Labels = ms.labels_
    segmented_image = np.reshape(Labels, [image.shape[0], image.shape[1]])

    plt.figure(1)
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig("E:/Suhas/Main/IUB/4th Sem/ML/Project/segmented.png")

    np_image_Lab = np.array(image_Lab)
    lum_img = np_image_Lab[:, :, 0]

    x1 = len(lum_img) / (X_points + 1)
    y1 = len(lum_img) / (Y_points + 1)
    features = []
    Labels = []
    for i in range(X_points):
        for j in range(Y_points):
            x = (i + 1) * x1
            y = (j + 1) * y1
            neighbors = neighbor(lum_img,x,y)
            DCT = dct(dct(neighbors.T, norm='ortho').T, norm='ortho')
            feature = np.reshape(DCT, -1).tolist()
            features.append(feature)
            pixel_Lab = image_Lab[int(x)][int(y)]
            pixel_Lab[0] = segmented_image[int(x)][int(y)]
            Labels.append(pixel_Lab)
    return np.array(features), np.array(Labels)


def PCA(data, n=3, largest=True):
    meanVals = np.mean(data, axis=0)
    meanRemoved = data - meanVals
    stdVals = np.std(meanRemoved, axis=0)
    normedData = meanRemoved / stdVals

    covMat = np.cov(normedData, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigVals_best = np.argsort(eigVals)

    #retain the eigenvalues and eigenvectors which we want
    i = -(n + 1) if largest else n
    j = -1 if largest else 1
    eigVals_best = eigVals_best[:i:j]
    eigVects_best = eigVects[:, eigVals_best]

    def feature_space(x):
        return ((x - meanVals) / stdVals) * eigVects_best

    return feature_space


def difference_calculation(data, label_dict, flag=True):
    n = len(label_dict)
    num_diffs = 500
    diff = np.zeros((num_diffs, np.shape(data)[1]))
    for i in range(0, num_diffs):
        l1 = randint(0, n - 1)
        l2 = l1
        if not flag:
            while True:
                l2 = randint(0, n - 1)
                if l1 != l2:
                    break

        j = randint(0, len(label_dict[l1]) - 1)
        k = randint(0, len(label_dict[l2]) - 1)
        diff[i, :] = data[j, :] - data[k, :]

    return diff


def feature_select(dct, labels):
    label_dict = {}
    for i in range(0, len(labels)):
        if labels[i] in label_dict:
            label_dict[labels[i]].append(i)
        else:
            label_dict[labels[i]] = [i]

    intra_diffs = difference_calculation(dct, label_dict, True)
    intra_diff_transform = PCA(intra_diffs, 40, False)
    inter_diffs = difference_calculation(intra_diff_transform(dct), label_dict, False)
    inter_diff_transform = PCA(inter_diffs, 10, True)

    def feature_trans(x):
        return inter_diff_transform(intra_diff_transform(x))

    return feature_trans