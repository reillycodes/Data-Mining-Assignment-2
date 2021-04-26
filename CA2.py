import numpy as np
import classKM
import matplotlib.pyplot as plt
class KMeans():
    def __init__(self, k, seed=7):
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

        test_data = np.copy(data)
        test_data = np.delete(test_data, -1,1)


        while self.convergance.any() != 0:
            for i in range(len(data)):
                for j in range(len(self.centroids)):
                    self.distances[j] = self.distance(test_data[i], self.centroids[j])
                cluster = np.argmin(self.distances)
                self.labels[i] = cluster

            self.old_centroids = np.copy(self.centroids)

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

            self.old_centroids = np.copy(self.centroids)

            for i in range(self.k):
                points = np.array([test_data[j] for j in range(len(test_data)) if self.labels[j] == i])
                self.centroids[i] = np.median(points, axis=0)

            for i in range(len(self.convergance)):
                self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        return self.labels

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


def l2_norm(data):
    norm_vector = np.linalg.norm(data, axis=0)

    normalised_data = data / norm_vector
    return normalised_data


def bscores(data, k, cluster_labels):

    global_precision = []
    global_recall = []
    global_fscore = []

    cluster_data = np.copy(data)

    cluster_data = np.column_stack((cluster_data, cluster_labels))


    lbls = np.unique(cluster_data[:, [-2]])

    precision_score = []
    recall_score = []
    fscore_score = []
    for i in range(k):
        c = []
        for j in range(len(cluster_data)):
            if cluster_data[j][-1] == i:
                c.append(cluster_data[j])
        c = np.array(c)
        for lbl in lbls:
            target = 0
            for i in c:
                if i[-2] == lbl:
                    target += 1
            total_labels= 0
            for l in cluster_data:
                if l[-2] == lbl:
                    total_labels += 1

            for q in range(target):
                precision = target / len(c)
                precision_score.append(precision)
                recall = target / total_labels
                recall_score.append(recall)
                fscore = 2 * precision * recall / (precision + recall)
                fscore_score.append(fscore)

    global_precision.append(np.mean(precision_score))

    global_recall.append(np.mean(recall_score))

    global_fscore.append((np.mean(fscore_score)))

    return [global_precision, global_recall, global_fscore]

animals = load_data('CA2data/animals')

countries = load_data('CA2data/countries')

fruits = load_data('CA2data/fruits')

veggies = load_data('CA2data/veggies')


data = np.vstack((animals, countries))

data = np.vstack((data,fruits))

data = np.vstack((data,veggies))


norm = np.copy(data)
norm = l2_norm(norm)

plot_precision = []
plot_recall = []
plot_fscore = []

plot_norm_precision = []
plot_norm_recall = []
plot_norm_fscore = []



for i in range(1,10):

    mean = KMeans(k=i)
    mean.fit(data)
    score = bscores(data,i,mean.labels)
    plot_precision.append(score[0])
    plot_recall.append(score[1])
    plot_fscore.append(score[2])

    mean_norm = classKM.KMeans(k=i)
    mean_norm.fit(norm)
    norm_score = bscores(norm,i,mean_norm.labels)
    plot_norm_precision.append(norm_score[0])
    plot_norm_recall.append(norm_score[1])
    plot_norm_fscore.append(norm_score[2])

for i in range(1,10):

    median = KMedians(k=i)
    median.fit(data)
    score = bscores(data, i, median.labels)
    plot_precision.append(score[0])
    plot_recall.append(score[1])
    plot_fscore.append(score[2])

    median_norm = classKM.KMedians(k=i)
    median_norm.fit(norm)
    norm_score = bscores(norm,i,median_norm.labels)
    plot_norm_precision.append(norm_score[0])
    plot_norm_recall.append(norm_score[1])
    plot_norm_fscore.append(norm_score[2])




plt.plot([k for k in range(1,10)],plot_precision[0:9], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_recall[0:9], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_fscore[0:9], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMeans')
plt.tight_layout()
plt.show()

plt.plot([k for k in range(1,10)],plot_precision[9:], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_recall[9:], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_fscore[9:], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMedians')
plt.tight_layout()
plt.show()

plt.plot([k for k in range(1,10)],plot_norm_precision[0:9], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_norm_recall[0:9], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_norm_fscore[0:9], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMeans - Normalised')
plt.tight_layout()
plt.show()

plt.plot([k for k in range(1,10)],plot_norm_precision[9:], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_norm_recall[9:], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_norm_fscore[9:], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMedians - Normalised')
plt.tight_layout()
plt.show()
