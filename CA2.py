import numpy as np
from copy import deepcopy
import bcubed
import classKM



def load_data(fname):
    features = []

    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ')
            del p[0]
            p = [float(i) for i in p]
            features.append(p)
    features = np.array(features)
    if fname == 'CA2data/animals':
        features = np.column_stack((features,[0]*features.shape[0]))
    elif fname == 'CA2data/countries':
        features = np.column_stack((features,[1]*features.shape[0]))
    elif fname == 'CA2data/fruits':
        features = np.column_stack((features,[2]*features.shape[0]))
    elif fname == 'CA2data/veggies':
        features = np.column_stack((features,[3]*features.shape[0]))
    return features

animals = load_data('CA2data/animals')
# print(animals)
countries = load_data('CA2data/countries')
# print(countries)
fruits = load_data('CA2data/fruits')
# print(fruits)
veggies = load_data('CA2data/veggies')
# print(veggies)

data = np.vstack((animals, countries))
data = np.vstack((data,fruits))
data = np.vstack((data,veggies))
print(data.shape)
model = classKM.KMeans(k=4)
model.fit(data)
# print(model.labels)

model2 = classKM.KMedians(k=3)
model2.fit(data)


cluster_data = deepcopy(data)
cluster_data = np.column_stack((cluster_data,model.labels))
print(cluster_data.shape)
lbls = np.unique(cluster_data[:,[-2]])

precision_score = []
recall_score = []
fscore_score = []
for i in range(4):
    c = []
    for j in range(len(cluster_data)):
        if cluster_data[j][-1] == i:
            c.append(cluster_data[j])
    c = np.array(c)
    for lbl in lbls:
        target = np.count_nonzero(np.where(c[:,[-2]]==lbl))
        for q in range(target):
            precision = target / len(c)
            precision_score.append(precision)
            recall = target / np.count_nonzero(np.where(cluster_data[:, -2] == lbl))
            recall_score.append(recall)
            fscore = 2 * precision * recall / (precision + recall)
            fscore_score.append(fscore)

pp = float(np.mean(precision_score))

rr =float(np.mean(recall_score))

ff = float((np.mean(fscore_score)))

print('%.3f' %pp)
print('%.3f' %rr)
print('%.3f' %ff)


# K is currently 4