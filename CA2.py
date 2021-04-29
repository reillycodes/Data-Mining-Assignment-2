import numpy as np
import matplotlib.pyplot as plt

#KMeans Class
class KMeans():
    '''
    KMeans Clustering Algorithm

    Parameters
    ---------
    k : int
        number of clusters to initialise
    seed: int
        random seed for reproducibility
    centroids: np.array
        starting number of centroids
    old_centroids: np.array
        empty array to use for convergence
    distances: np.array
        array used to measure distance from datapoint to centroid
    labels: np.array
        array to track currently assigned cluster of datapoint
    convergence: np.array
        used to check if convergence has been met
    '''
    def __init__(self, k, seed=42):
        self.k = k
        self.seed = seed
        self.centroids = np.empty(self.k)
        self.old_centroids = np.empty(self.k)
        self.distances = np.empty(self.k)
        self.labels = np.empty(self.k)
        self.convergence = np.empty(self.k)

    '''
    Creates random starting points for the centroids from the dataset.
    Parameters
    ---------
    data : np.array
        dataset to make random point from
    '''
    def gen_random_points(self,data):

        min_data = np.min(data, axis=0) #axis 0 finds min from features not row
        max_data = np.max(data, axis=0) #axis 0 finds max from features not row
        random_points = np.random.uniform(low=min_data, high=max_data)
        return random_points
    '''
    Used to calculate Euclidean Distance
    Parameters
    ----------
    x: np.array
        Dataset
    y: np.array
        Centroid
    '''
    def distance(self,x,y):

        return np.linalg.norm(x - y)
    '''
    Uses the dataset to create clusters based on the number of Ks
    Parameters
    ----------
    data: np.array
        data used for clustering
    '''
    def fit(self,data):

        np.random.seed(self.seed)
        self.centroids = np.zeros((self.k, data.shape[1])) #initalise centroids with number of features in dataset

        for i in range(self.k):
            self.centroids[i] = self.gen_random_points(data) # create centroid starting points

        #initilaise all arrays to correct shapes needed
        self.old_centroids = np.zeros(self.centroids.shape[0])
        self.distances = np.zeros(self.centroids.shape[0])
        self.labels = np.zeros((len(data)))
        self.convergence = np.zeros(self.centroids.shape[0])

        #initalises the convergence measurements
        for i in range(len(self.convergence)):
            self.convergence[i] = self.distance(self.centroids[i], self.old_centroids[i])

        #copy data to use within model to ensure no unexpected edits to original data
        test_data = np.copy(data)

        while self.convergence.any() != 0: #algorithm will continue till convergence

            # Find distance for each datapoint to each centroid and assign cluster with smallest distance to datapoint
            for i in range(len(data)):
                for j in range(len(self.centroids)):
                    self.distances[j] = self.distance(test_data[i], self.centroids[j])
                cluster = np.argmin(self.distances)
                self.labels[i] = cluster

            # Keep centroid position before update
            self.old_centroids = np.copy(self.centroids)

            # Find mean values of each feature within cluster and set the means as the new centroid position
            for i in range(self.k):
                points = np.array([test_data[j] for j in range(len(test_data)) if self.labels[j] == i])
                self.centroids[i] = np.mean(points, axis=0)

            # Find distance between new centroid positions and previous, if no convergence then alogrithm goes again
            # with new centroid positions
            for i in range(len(self.convergence)):
                self.convergence[i] = self.distance(self.centroids[i], self.old_centroids[i])

        return self.labels

#KMedians Class
class KMedians():
    '''
    KMedians Clustering Algorithm

    Parameters
    ---------
    k : int
        number of clusters to initialise
    seed: int
        random seed for reproducibility
    centroids: np.array
        starting number of centroids
    old_centroids: np.array
        empty array to use for convergence
    distances: np.array
        array used to measure distance from datapoint to centroid
    labels: np.array
        array to track currently assigned cluster of datapoint
    convergence: np.array
        used to check if convergence has been met
    '''
    def __init__(self, k, seed=42):
        self.k = k
        self.seed = seed
        self.centroids = np.empty(self.k)
        self.old_centroids = np.empty(self.k)
        self.distances = np.empty(self.k)
        self.labels = np.empty(self.k)
        self.convergance = np.empty(self.k)

    '''
    Creates random starting points for the centroids from the dataset.
    Parameters
    ---------
    data : np.array
        dataset to make random point from
    '''
    def gen_random_points(self,data):

        min_data = np.min(data, axis=0)
        max_data = np.max(data, axis=0)
        random_points = np.random.uniform(low=min_data, high=max_data)
        return random_points

    '''
    Used to calculate Manhattan Distance
    Parameters
    ----------
    x: np.array
        Dataset
    y: np.array
        Centroid
    '''
    def distance(self,x,y):

        return np.sum(np.abs(x-y))

    '''
    Uses the dataset to create clusters based on the number of Ks
    Parameters
    ----------
    data: np.array
        data used for clustering
    '''
    def fit(self,data):

        np.random.seed(self.seed)
        self.centroids = np.zeros((self.k, data.shape[1])) #initalise centroids with number of features in dataset

        for i in range(self.k):
            self.centroids[i] = self.gen_random_points(data) # create centroid starting points

        # initialise all arrays to correct shapes needed
        self.old_centroids = np.zeros(self.centroids.shape[0])
        self.distances = np.zeros(self.centroids.shape[0])
        self.labels = np.zeros((len(data)))
        self.convergance = np.zeros(self.centroids.shape[0])

        #initalises the convergence measurements
        for i in range(len(self.convergance)):
            self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        #copy data to use within model to ensure no unexpected edits to original data
        test_data = np.copy(data)

        while self.convergance.any() != 0: #algorithm will continue till convergence

            # Find distance for each datapoint to each centroid and assign cluster with smallest distance to datapoint
            for i in range(len(data)):
                for j in range(len(self.centroids)):
                    self.distances[j] = self.distance(test_data[i], self.centroids[j])
                cluster = np.argmin(self.distances)
                self.labels[i] = cluster

            # Keep centroid position before update
            self.old_centroids = np.copy(self.centroids)

            # Find median values of each feature within cluster and set the medians as the new centroid position
            for i in range(self.k):
                points = np.array([test_data[j] for j in range(len(test_data)) if self.labels[j] == i])
                self.centroids[i] = np.median(points, axis=0)

            # Find distance between new centroid positions and previous, if no convergence then alogrithm goes again
            # with new centroid positions
            for i in range(len(self.convergance)):
                self.convergance[i] = self.distance(self.centroids[i], self.old_centroids[i])

        return self.labels

# Import Data function
def load_data(fname):
    features = []

    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ') #strip white space
            del p[0] #delete the string at the beginning of each row
            p = [float(i) for i in p] # convert to float
            features.append(p)
    features = np.array(features) # convert each row list into an array
    # Appends a label to each datapoint to classify where it came from originally
    # This will be used for the bcubed calculations
    if fname == 'CA2data/animals':
        features = np.column_stack((features,[0]*features.shape[0]))
    elif fname == 'CA2data/countries':
        features = np.column_stack((features,[1]*features.shape[0]))
    elif fname == 'CA2data/fruits':
        features = np.column_stack((features,[2]*features.shape[0]))
    elif fname == 'CA2data/veggies':
        features = np.column_stack((features,[3]*features.shape[0]))
    return features

# Function to combine seperate datafiles into one, create classification array and normalise data
def make_data(data1,data2,data3,data4):
    data = np.vstack((data1, data2))
    data = np.vstack((data, data3))
    data = np.vstack((data, data4))
    classes = np.copy(data[:, -1])
    data = np.delete(data,-1, 1)
    norm = np.copy(data)
    norm = l2_norm(norm)

    return data, classes, norm

#Normalise data
def l2_norm(data):
    # Normalise each vector to unit length l2
    return data/(np.linalg.norm(data,axis=1,keepdims=True))

'''
BCUBED Function
Used to work out BCUBED Precsion, Recall, FScore
Parameters
------------

data: np.array
    Dataset used for clustering
k: int
    number of clusters
class_labels: np.array
    Classification of datapoints from original data
cluster_labels: np.array
    Cluster each datapoint belongs to after running through algorithm
    
Returns:
BCUBED Precision, Recall, F-Score for each model
'''
def bscores(data, k, class_labels, cluster_labels):

    # Data and labels merged into one dataset
    cluster_data = np.copy(data)
    cluster_data = np.column_stack((cluster_data,class_labels))
    cluster_data = np.column_stack((cluster_data, cluster_labels))

    #Initalise score lists
    precision_score = []
    recall_score = []
    fscore_score = []

    # Create an array of each unique classification from dataset
    classification_label = np.unique(cluster_data[:, [-2]])

    #BCUBED Score calculations
    for i in range(k):
        # Get all datapoints for each cluster
        current_cluster = []
        for j in range(len(cluster_data)):
            if cluster_data[j][-1] == i:
                current_cluster.append(cluster_data[j])
        current_cluster = np.array(current_cluster)
        #For each label iterates through cluster
        for label in classification_label:
            #target is equal to number of items in cluster of label in classification_label
            target = np.count_nonzero(current_cluster[:,-2]==label)
            # Iterates target number of times to add each precision, recall and fscore for each datapoint and appends
            # score to the respective list
            for q in range(target):
                precision = target / len(current_cluster) # No. of items in C(x) with A(x) / No. of items C(x)
                precision_score.append(precision)
                recall = target / np.count_nonzero(cluster_data[:, -2] == label) # No. of items in C(x) with A(x)/ Total no. of times with A(x)
                recall_score.append(recall)
                fscore = (2 * precision * recall)/ (precision + recall)
                fscore_score.append(fscore)

    # Return the mean of each score list to get final score
    return [np.mean(precision_score), np.mean(recall_score), np.mean(fscore_score)]

animals = load_data('CA2data/animals')
countries = load_data('CA2data/countries')
fruits = load_data('CA2data/fruits')
veggies = load_data('CA2data/veggies')

#Create dataset, classes, and normalised data
data, classes, norm = make_data(animals,countries,fruits,veggies)

# Lists used for plotting scores
plot_scores = [[],[],[]]
plot_norm_scores = [[],[],[]]




#Code used to run each model, uncomment if needed
#KMeans
for i in range(1,10):

    mean = KMeans(k=i)
    mean.fit(data)
    score = bscores(data,i,classes,mean.labels)
    print('KMeans')
    print('K =',i, 'BCUBED scores:\n\nPrecision:',score[0],'\nRecall:',score[1],'\nFScore:',score[2],'\n')
    plot_scores[0].append(score[0])
    plot_scores[1].append(score[1])
    plot_scores[2].append(score[2])

#KMeans Normlaised
for i in range(1,10):
    mean_norm = KMeans(k=i)
    mean_norm.fit(norm)
    norm_score = bscores(norm,i,classes, mean_norm.labels)
    print('KMeans Normalised')
    print('K =', i, 'BCUBED scores:\n\nPrecision:', score[0], '\nRecall:', score[1], '\nFScore:', score[2], '\n')
    plot_norm_scores[0].append(norm_score[0])
    plot_norm_scores[1].append(norm_score[1])
    plot_norm_scores[2].append(norm_score[2])
#KMedians
for i in range(1,10):

    median = KMedians(k=i)
    median.fit(data)
    score = bscores(data, i, classes,median.labels)
    print('Kmedians')
    print('K =',i, 'BCUBED scores:\n\nPrecision:',score[0],'\nRecall:',score[1],'\nFScore:',score[2],'\n')
    plot_scores[0].append(score[0])
    plot_scores[1].append(score[1])
    plot_scores[2].append(score[2])
#KMedians Normalised
for i in range(1,10):
    median_norm = KMedians(k=i)
    median_norm.fit(norm)
    norm_score = bscores(norm,i,classes,median_norm.labels)
    print('KMedians Normalised')
    print('K =',i, 'BCUBED scores:\n\nPrecision:',score[0],'\nRecall:',score[1],'\nFScore:',score[2],'\n')
    plot_norm_scores[0].append(norm_score[0])
    plot_norm_scores[1].append(norm_score[1])
    plot_norm_scores[2].append(norm_score[2])



# Code used to create charts, uncomment to run if needed

plt.plot([k for k in range(1,10)],plot_scores[0][0:9], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_scores[1][0:9], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_scores[2][0:9], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMeans')
plt.tight_layout()
plt.savefig('Kmeans.png')
plt.show()

plt.plot([k for k in range(1,10)],plot_scores[0][9:], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_scores[1][9:], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_scores[2][9:], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMedians')
plt.tight_layout()
plt.savefig('Kmedians.png')
plt.show()

plt.plot([k for k in range(1,10)],plot_norm_scores[0][0:9], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_norm_scores[1][0:9], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_norm_scores[2][0:9], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMeans - Normalised')
plt.tight_layout()
plt.savefig('Kmeans Normalised.png')
plt.show()

plt.plot([k for k in range(1,10)],plot_norm_scores[0][9:], label = 'Precision')
plt.plot([k for k in range(1,10)], plot_norm_scores[1][9:], label = 'Recall')
plt.plot([k for k in range(1,10)], plot_norm_scores[2][9:], label = 'F-Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('BCubed Score')
plt.title('KMedians - Normalised')
plt.tight_layout()
plt.savefig('Kmedians Normalised.png')
plt.show()
