'''
Arlind Stafaj
Data Mining 6930
Spring 2020
Homework # 4 - Question 1
'''

import pandas as pd
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import statistics

img = skimage.io.imread('image.png')
skimage.io.imshow(img)
plt.show()
print(img.shape)
k = [2, 3, 6, 10]

centroids = [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [
    0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9]]

nimage = img/255
clusters = [[60, 179, 113], [0, 191, 255], [255, 255, 0], [255, 0, 0], [0, 0, 0], [
    169, 169, 169], [255, 140, 0], [128, 0, 128], [255, 192, 203], [255, 255, 255]]


class K_Means:
    def __init__(self, k=2, num_iters=50):
        self.k = k
        self.num_iters = num_iters

    # predict image colors
    def predict(self, X, centroids, clusters):
        self.X = X
        self.centroids = centroids[:self.k]
        self.clusters = clusters[:self.k]

        finalPoints = []
        for n in range(self.num_iters):
            print('numer of iterations before convergence: ', n)
            numOfCluster = []
            newCentroids = []
            edCluster = []
            for x in self.X:
                num, ed = self._cluster_groups(x, self.centroids)
                numOfCluster.append(num)
                edCluster.append(ed)

            newCentroids = self._getNewCentroids(numOfCluster, self.X)
            if np.array_equal(newCentroids, self.centroids):
                break
            else:
                self.centroids = newCentroids
            finalPoints = numOfCluster
        print('\n')
        finalPoints = self._toPlot(numOfCluster, clusters)
        sse = np.sum(np.square(edCluster))
        return self.centroids, finalPoints, sse

    def _cluster_groups(self, x, centroids):
        # Euclidean distance of each data point - C(i)
        edRow = []

        # Holds number of the cluster a pixel belongs to
        clustersNum = []
        # Holds shortest Euclidean Distance's
        edClust = []

        for i in range(self.k):
            edClusterDist = np.sqrt(
                np.sum((np.square((x-centroids[i]))), axis=1))
            edRow.append(edClusterDist)

        temp = np.array(edRow).T
        temp = temp.tolist()
        for i in range(len(temp)):
            clustersNum.append(temp[i].index(min(temp[i])))
            edClust.append(min(temp[i]))

        return clustersNum, edClust

    def _getNewCentroids(self, numOfCluster, X):
        clusters = [[] for _ in range(self.k)]
        for x in range(X.shape[0]):
            for i in range(X.shape[1]):
                clusters[numOfCluster[x][i]].append(X[x][i])

        newCentroids = []
        for i in range(len(clusters)):
            newCentroids.append(np.mean(clusters[i], axis=0))

        return newCentroids

    def _toPlot(self, numOfCluster, clusters):
        newX = [[el] for el in numOfCluster]
        for i in range(self.k):
            for n in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    if newX[n][0][j] == i:
                        newX[n][0][j] = clusters[i]

        newX = np.array(newX).reshape(
            self.X.shape[0], self.X.shape[1], self.X.shape[2])
        return newX


sseList = []
for i in k:
    print('Printing results for k=', i)
    test = K_Means(k=i, num_iters=50)
    centroid, im, sse = test.predict(nimage, centroids, clusters)
    print('centroids =', centroid)
    print('-------------------------------------------------\n',
          'Sum of Squared Error:', sse, '\n')
    sseList.append(sse)
    skimage.io.imshow(np.array(im))
    plt.show()

for i in range(len(k)):
    print('k =', k[i], 'SSE = ', sseList[i])
print('\n')

plt.title('SSE vs K')
plt.xlabel('k')
plt.ylabel('SSE')
plt.grid(True)
plt.tight_layout()
plt.plot(k, sseList, marker='o')
plt.show()
