import sys
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from similarity_class import Similarity
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k = 2, max_iter = 200):
        self.__k = k
        self.__max_iter = max_iter

    def fit(self, dataset):
        self.__dataset = dataset
        self.__ndim = dataset.shape[1]
        self.centroids =  np.random.random((self.__k, self.__ndim))
        
        # Calculating the distance to centroid  
        for iter in range(self.__max_iter):

            self.classification_list = {}

            for i in range(self.__k):
                self.classification_list[i] = []
            
            for data in self.__dataset:
                distance = [Similarity(data, centroid).euclidean_distance() for centroid in self.centroids]
                classification = distance.index(min(distance))
                self.classification_list[classification].append(data)


            # Update
            for classification in self.classification_list:
                self.centroids[classification] = np.average(self.classification_list[classification],axis=0)            

        
        #
        return self.classification_list, self.centroids
    

    def predict(self,data):
        distances = [Similarity(data, centroid).euclidean_distance() for centroid in self.__centroids]
        classification = distances.index(min(distances))
        return classification


dataset = np.random.random((200,2))

kmean = Kmeans(k = 10, max_iter = 300)
dataset_labeled, centroids = kmean.fit(dataset)
