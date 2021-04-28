import numpy as np
import matplotlib.pyplot as plt

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
        return random_points

    def distance(self,x,y):
        return np.linalg.norm(x - y)

    def fit(self,data):

        np.random.seed(self.seed)
        self.centroids = np.zeros((self.k, data.shape[1]))

        for i in range(self.k):
            self.centroids[i] = self.gen_random_points(data)

        self.old_centroids = np.zeros(self.centroids.shape[0])
        self.distances = np.zeros(self.centroids.shape[0])
        self.labels = np.zeros((len(data)))
        self.convergance = np.zeros(self.centroids.shape[0])

        for i in range(len(self.convergance)):
            self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        test_data = np.copy(data)


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
        return random_points

    def distance(self,x,y):

        return np.sum(np.abs(x-y))

    def fit(self,data):
        np.random.seed(self.seed)
        self.centroids = np.zeros((self.k, data.shape[1]))

        for i in range(self.k):
            self.centroids[i] = self.gen_random_points(data)

        self.old_centroids = np.zeros(self.centroids.shape[0])
        self.distances = np.zeros(self.centroids.shape[0])
        self.labels = np.zeros((len(data)))
        self.convergance = np.zeros(self.centroids.shape[0])

        for i in range(len(self.convergance)):
            self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        test_data = np.copy(data)

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

# def l2_norm(data):
#
#     for i in data:
#         x = np.linalg.norm(i)
#         for j in range(0,len(i)):
#             i[j] = i[j]/x
#
#     return data

def l2_norm(data):

    return data/(np.linalg.norm(data,axis=1,keepdims=True))

def bscores(data, k, class_labels, cluster_labels):

    cluster_data = np.copy(data)
    cluster_data = np.column_stack((cluster_data,class_labels))
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

    return [np.mean(precision_score), np.mean(recall_score), np.mean(fscore_score)]

def make_data(data1,data2,data3,data4):
    data = np.vstack((data1, data2))
    data = np.vstack((data, data3))
    data = np.vstack((data, data4))
    classes = np.copy(data[:, -1])
    data = np.delete(data, np.s_[-1], 1)
    norm = np.copy(data)
    norm = l2_norm(norm)

    return data, classes, norm

animals = load_data('CA2data/animals')
countries = load_data('CA2data/countries')
fruits = load_data('CA2data/fruits')
veggies = load_data('CA2data/veggies')

data, classes, norm = make_data(animals,countries,fruits,veggies)

plot_scores = [[],[],[]]
plot_norm_scores = [[],[],[]]
seed=42
for i in range(1,10):

    mean = KMeans(k=i,seed=seed)
    mean.fit(data)
    score = bscores(data,i,classes,mean.labels)
    # print('KMeans')
    # print('K =',i, 'BCUBED scores:\n\nPrecision:',score[0],'\nRecall:',score[1],'\nFScore:',score[2],'\n')
    plot_scores[0].append(score[0])
    plot_scores[1].append(score[1])
    plot_scores[2].append(score[2])

for i in range(1,10):
    mean_norm = KMeans(k=i)
    mean_norm.fit(norm)
    norm_score = bscores(norm,i,classes, mean_norm.labels)
    # print('KMeans Normalised')
    # print('K =', i, 'BCUBED scores:\n\nPrecision:', score[0], '\nRecall:', score[1], '\nFScore:', score[2], '\n')
    plot_norm_scores[0].append(norm_score[0])
    plot_norm_scores[1].append(norm_score[1])
    plot_norm_scores[2].append(norm_score[2])

for i in range(1,10):

    median = KMedians(k=i,seed=seed)
    median.fit(data)
    score = bscores(data, i, classes,median.labels)
    # print('Kmedians')
    # print('K =',i, 'BCUBED scores:\n\nPrecision:',score[0],'\nRecall:',score[1],'\nFScore:',score[2],'\n')
    plot_scores[0].append(score[0])
    plot_scores[1].append(score[1])
    plot_scores[2].append(score[2])

for i in range(1,10):
    median_norm = KMedians(k=i)
    median_norm.fit(norm)
    norm_score = bscores(norm,i,classes,median_norm.labels)
    # print('KMedians Normalised')
    # print('K =',i, 'BCUBED scores:\n\nPrecision:',score[0],'\nRecall:',score[1],'\nFScore:',score[2],'\n')
    plot_norm_scores[0].append(norm_score[0])
    plot_norm_scores[1].append(norm_score[1])
    plot_norm_scores[2].append(norm_score[2])


plot_graphs = True

if plot_graphs == True:

    plt.plot([k for k in range(1,10)],plot_scores[0][0:9], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_scores[1][0:9], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_scores[2][0:9], label = 'F-Score')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMeans')
    plt.tight_layout()
    plt.show()

    plt.plot([k for k in range(1,10)],plot_scores[0][9:], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_scores[1][9:], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_scores[2][9:], label = 'F-Score')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMedians')
    plt.tight_layout()
    plt.show()

    plt.plot([k for k in range(1,10)],plot_norm_scores[0][0:9], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_norm_scores[1][0:9], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_norm_scores[2][0:9], label = 'F-Score')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMeans - Normalised')
    plt.tight_layout()
    plt.show()

    plt.plot([k for k in range(1,10)],plot_norm_scores[0][9:], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_norm_scores[1][9:], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_norm_scores[2][9:], label = 'F-Score')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMedians - Normalised')
    plt.tight_layout()
    plt.show()
