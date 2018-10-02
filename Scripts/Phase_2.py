import numpy as np
from sklearn import neighbors
import math
sampling_side = 7
X_points = 100
Y_points = 100
n = 7
max_conf_val = 0.0

def neighbor(image, x, y, size = sampling_side):
	n = (size-1)/2
	x_min = x - n
	x_max = x + n + 1
	y_min = y - n
	y_max = y + n + 1
	neighbors = image[x_min : x_max, y_min : y_max]
	return neighbors

def knn(test, train, target):
    features = []
    dist = []
    ctrain = []

    classifier = neighbors.KNeighborsClassifier(n)
    classifier.fit(train, target)

    for i in range(0, np.shape(test)[0]):
        if i % 10000 == 0:
            print "Running iteration ", i

        ptarget = classifier.predict(test[i, :])
        d, n1 = classifier.kneighbors(test[i, :])

        index = -1
        max = np.max(d) + 1
        for j in range(0, n):
            if d[0][j] < max and target[n1[0][j]] == ptarget:
                max = d[0][j]
                index = n1[0][j]

        features.append(ptarget)
        dist.append(max)
        ctrain.append(index)

    return features, dist, ctrain


def neighbor_calculations(w, x, y, neighbors, dist):
    for i in np.nditer(neighbors):
        w[x][y] += math.exp(-i)
    w[x][y] = math.exp(-dist[x][y]) / w[x][y]
    return w[x][y]


def weight_calculate(dist):
    X = len(dist)
    Y = len(dist[0])
    n = (sampling_side - 1) / 2
    w = np.zeros(dist.shape)
    i = 0

    for x in range(n, X - n):
        for y in range(n, Y - n):
            if i % 5000 == 0:
                print "Running iteration ", i
            i += 1
            neighbors = neighbor(dist, x, y)
            w[x][y] = neighbor_calculations(w, x, y, neighbors, dist)

    return w[n:X-n, n:Y-n]


def confidence_level_calculations(labels, neighb_labels, neighb_weights, d, max_conf, max_label):
    for k in labels:
        num = 0
        for (i, j), n_label in np.ndenumerate(neighb_labels):
            if (k == n_label):
                num += neighb_weights[i][j]
        if(num / d > max_conf):
            max_conf = num / d
            max_label = k
    return max_conf, max_label


def weights_sum_calculation(neighb_labels, dom_labels, colors, x, y, neighb_weights, train_labels, neighb_closest_train):
    weight_sum = 0
    for (i, j), n_label in np.ndenumerate(neighb_labels):
        if (dom_labels[x][y] == n_label):
            colors[x][y] += neighb_weights[i][j] * train_labels[neighb_closest_train[i][j]]
            weight_sum += neighb_weights[i][j]
    colors[x][y] /= weight_sum
    return colors


def voting_of_image(features, weights):
    X = len(weights)
    Y = len(weights[0])
    n = (sampling_side-1)/2
    dtarget = np.zeros(weights.shape)
    conf = np.zeros(weights.shape)
    i=0

    for x in range(n, X-n):
        for y in range(n, Y-n):
            if i % 5000 == 0:
                print "Running iteration ", i
            i += 1
            neighb_labels = neighbor(features, x+n, y+n)
            neighb_weights = neighbor(weights, x, y)
            #labels = np.unique(neighb_labels.flatten())
            labels = set(np.reshape(neighb_labels, -1))
            max_conf = -1
            max_label = -1
            #den = np.sum(neighb_weights)
            max_conf, max_label = confidence_level_calculations(labels, neighb_labels, neighb_weights, np.sum(neighb_weights), max_conf, max_label)
            dtarget[x, y] = max_label
            conf[x, y] = max_conf
    return dtarget[n:X-n, n:Y-n], conf[n:X-n, n:Y-n]


def color_optimize(dtarget, conf, weights, train, train_labels):
    #X, Y = dtarget.shape
    X = len(dtarget)
    Y = len(dtarget[0])
    n = (sampling_side - 1) / 2
    colors = np.zeros(dtarget.shape + (2,))
    i = 0
    final_pixels = []

    for x in range(n, X-n):
        for y in range(n, Y-n):
            if i % 5000 == 0:
                print "Running iteration ", i
            i += 1

            if (conf[x][y] < max_conf_val):
                continue

            neighb_labels = neighbor(dtarget, x, y)
            neighb_weights = neighbor(weights, x+n, y+n)
            neighb_closest_train = neighbor(train, x + 2*n, y + 2*n)

            colors = weights_sum_calculation(neighb_labels, dtarget, colors, x, y, neighb_weights, train_labels, neighb_closest_train)
            final_pixels.append([x + 3*n, y + 3*n, colors[x][y][0], colors[x][y][1]])

    return colors[n:-n, n:-n], np.array(final_pixels)