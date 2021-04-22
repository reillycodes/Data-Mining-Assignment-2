import numpy as np
from scipy.spatial import distance
# from copy import deepcopy
#
#
# def distance(x,y):
#     return np.linalg.norm(x-y)
#
# x = np.array([[2,5,0],[5,6,1],[4,6,1],[5,5,2],[12,10,2]])
# centroids = np.array([[1,3],[4,4],[7,9]])
# old_centroids = np.zeros(centroids.shape[0])
# k =3
# # print(cluster_classification(x))
# distances = np.zeros(centroids.shape[0])
# labels = np.zeros(len(x))
#
# error = np.zeros(centroids.shape[0])
# #Assign datapoints to nearest cluster
# for i in range(len(error)):
#     error[i] = distance(centroids[i], old_centroids[i])
# while error.all() != 0:
#     for i in range(len(x)):
#         for j in range(len(centroids)):
#             distances[j] = distance(x[i][0:2],centroids[j])
#         cluster=np.argmin(distances)
#         labels[i] = cluster
#
#     # print(labels)
#
#     #keep track of previous centroids to move values to
#     centroids_old = deepcopy(centroids)
#
#     # update centroids
#
#     for i in range(k):
#         points = np.array([x[j] for j in range(len(x)) if labels[j] == i])
#         centroids[i] = np.mean(points[:,[0,1]], axis=0)
#
#     for i in range(len(error)):
#         error[i] = distance(centroids[i],centroids_old[i])
#
#     print(labels)
#
# import numpy as np
# from copy import deepcopy
# import classKM
# def load_data(fname):
#     features = []
#
#     with open(fname) as F:
#         for line in F:
#             p = line.strip().split(' ')
#             del p[0]
#             p = [float(i) for i in p]
#             features.append(p)
#     features = np.array(features)
#     if fname == 'CA2data/animals':
#         features = np.column_stack((features,[0]*features.shape[0]))
#     elif fname == 'CA2data/countries':
#         features = np.column_stack((features,[1]*features.shape[0]))
#     elif fname == 'CA2data/fruits':
#         features = np.column_stack((features,[2]*features.shape[0]))
#     elif fname == 'CA2data/veggies':
#         features = np.column_stack((features,[3]*features.shape[0]))
#     return features
#
# animals = load_data('CA2data/animals')
# # print(animals)
# countries = load_data('CA2data/countries')
# # print(countries)
# fruits = load_data('CA2data/fruits')
# # print(fruits)
# veggies = load_data('CA2data/veggies')
# # print(veggies)
# np.random.seed(42)
# def gen_random_points(data):
#     min_data = np.min(data, axis=0)
#     max_data = np.max(data, axis=0)
#     random_points=np.random.uniform(low=min_data, high=max_data)
#     random_points = list(random_points)
#     del random_points[300]
#     random_points = np.array(random_points)
#     return random_points
#
# def edistance(x,y):
#     return np.linalg.norm(x-y)
#
# k = 4
#
# data = np.vstack((animals, countries))
# data = np.vstack((data,fruits))
# data = np.vstack((data,veggies))
# centroids = np.zeros((k,data.shape[1]))
# centroids= np.delete(centroids,-1,1)
#
# for i in range(k):
#     centroids[i] = gen_random_points(data)
#
# old_centroids = np.zeros(centroids.shape[0])
# distances = np.zeros(centroids.shape[0])
# labels = np.zeros((len(data)))
# convergance = np.zeros(centroids.shape[0])
#
# for i in range(len(convergance)):
#     convergance[i] = edistance(centroids[i],old_centroids[i])
#
# test_data = np.delete(data,-1,1)
#
# while convergance.any() != 0:
#     for i in range(len(data)):
#         for j in range(len(centroids)):
#             distances[j] = edistance(test_data[i],centroids[j])
#         cluster = np.argmin(distances)
#         labels[i] = cluster
#
#     old_centroids = deepcopy(centroids)
#
#     for i in range(k):
#         points = np.array([test_data[j] for j in range(len(test_data)) if labels[j] == i])
#         centroids[i] = np.mean(points, axis=0)
#
#     for i in range(len(convergance)):
#         convergance[i] = edistance(centroids[i],old_centroids[i])
#
#     print(convergance)


def manhattan_distance(x,y):
    return np.sum(np.abs(x-y))

data = np.array([-3,4,5,6])
centroid = np.array([12,43,5,1])
print(np.abs(data))

print(manhattan_distance(data,centroid))
print(distance.cityblock(data,centroid))