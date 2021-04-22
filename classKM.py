import numpy as np
from copy import deepcopy

class KMeans():
    def __init__(self, k, seed=42):
        self.k = k
        self.seed = seed
        self.centroids = np.empty(self.k)
        self.old_centroids = np.empty(self.k)
        self.distances = np.empty(self.k)
        self.labels = np.empty(self.k)
        self.convergance = np.empty(self.k)

    def gen_random_points(self,data):

        min_data = np.min(data, axis=0)
        max_data = np.max(data, axis=0)
        random_points = np.random.uniform(low=min_data, high=max_data)
        random_points = list(random_points)
        del random_points[300]
        random_points = np.array(random_points)
        return random_points

    def distance(self,x,y):
        return np.linalg.norm(x - y)

    def fit(self,data):
        np.random.seed(self.seed)
        self.centroids = np.zeros((self.k, data.shape[1]))
        self.centroids = np.delete(self.centroids, -1, 1)

        for i in range(self.k):
            self.centroids[i] = self.gen_random_points(data)

        self.old_centroids = np.zeros(self.centroids.shape[0])
        self.distances = np.zeros(self.centroids.shape[0])
        self.labels = np.zeros((len(data)))
        self.convergance = np.zeros(self.centroids.shape[0])

        for i in range(len(self.convergance)):
            self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        test_data = np.delete(data, -1, 1)

        while self.convergance.any() != 0:
            for i in range(len(data)):
                for j in range(len(self.centroids)):
                    self.distances[j] = self.distance(test_data[i], self.centroids[j])
                cluster = np.argmin(self.distances)
                self.labels[i] = cluster

            self.old_centroids = deepcopy(self.centroids)

            for i in range(self.k):
                points = np.array([test_data[j] for j in range(len(test_data)) if self.labels[j] == i])
                self.centroids[i] = np.mean(points, axis=0)

            for i in range(len(self.convergance)):
                self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        return self.labels

class KMedians():
    def __init__(self, k, seed=42):
        self.k = k
        self.seed = seed
        self.centroids = np.empty(self.k)
        self.old_centroids = np.empty(self.k)
        self.distances = np.empty(self.k)
        self.labels = np.empty(self.k)
        self.convergance = np.empty(self.k)

    def gen_random_points(self,data):

        min_data = np.min(data, axis=0)
        max_data = np.max(data, axis=0)
        random_points = np.random.uniform(low=min_data, high=max_data)
        random_points = list(random_points)
        del random_points[300]
        random_points = np.array(random_points)
        return random_points

    def distance(self,x,y):

        return np.sum(np.abs(x-y))

    def fit(self,data):
        np.random.seed(self.seed)
        self.centroids = np.zeros((self.k, data.shape[1]))
        self.centroids = np.delete(self.centroids, -1, 1)

        for i in range(self.k):
            self.centroids[i] = self.gen_random_points(data)

        self.old_centroids = np.zeros(self.centroids.shape[0])
        self.distances = np.zeros(self.centroids.shape[0])
        self.labels = np.zeros((len(data)))
        self.convergance = np.zeros(self.centroids.shape[0])

        for i in range(len(self.convergance)):
            self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        test_data = np.delete(data, -1, 1)

        while self.convergance.any() != 0:
            for i in range(len(data)):
                for j in range(len(self.centroids)):
                    self.distances[j] = self.distance(test_data[i], self.centroids[j])
                cluster = np.argmin(self.distances)
                self.labels[i] = cluster

            self.old_centroids = deepcopy(self.centroids)

            for i in range(self.k):
                points = np.array([test_data[j] for j in range(len(test_data)) if self.labels[j] == i])
                self.centroids[i] = np.median(points, axis=0)

            for i in range(len(self.convergance)):
                self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        return self.labels



