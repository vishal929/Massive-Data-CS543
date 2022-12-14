# this file holds the logic for the CURE

# first step is to sample data from 1 pass
# 1) sample data and cluster heirarchically to find initial clusters
# 2) for each cluster pick r representative points, as dispersed as possible from each other in the sample
# 3) move the representative points p% closer to the centroid, say 20%
# final pass over dataset
# 4) pass over dataset and assign points to their nearest cluster based on representatives (tagging step)
# Note that step 4 is not related to creating the clusters, it is just assigning a tag. Not centroid is recomputed
# This tagging step can just be done in pyspark map reduce if memory allows, but we will also provide a tag function here for a single point


import numpy as np
import random
from sklearn.cluster import AgglomerativeClustering


class CURE_Cluster():
    
    # clusters have a centroid (from the initial heirarchical clustering step)

    # clusters have r representatives (which may not be assigned yet at initialization)

    # we initialize this cluster with a given state
    def __init__(self, centroid, r):
        self.centroid = centroid
        self.representatives = []
        self.r = r

    # we set r representatives for this cluster from the samples which were assigned to this cluster
    def set_representatives(self,sample):
        for i in range(self.r):
            if i ==0:
                # first point is chosen at random
                random_point = random.choice(sample)
                self.representatives.append(random_point)
            else: 
                # next point is the furthest distance from the previous points
                # distances is an array where distances[i] = distance of sample i from our current representatives
                distance_candidate = None
                for example in sample:
                    distance = 0
                    for representative in self.representatives:
                        distance += np.linalg.norm(example-representative)
                    if distance_candidate is None or distance > distance_candidate[0]:
                        distance_candidate = (distance,example)
                # picking the representative with the greatest sum of distances to our previous points
                self.representatives.append(distance_candidate[1])


    # we move the r representatives a fraction closer to the centroid
    def move_representatives(self,fraction):
        for i,representative in enumerate(self.representatives):
            # getting direction from representative to centroid and the distance
            difference= np.subtract(self.centroid,representative)
            magnitude = np.linalg.norm(difference)
            if magnitude !=0:
                direction = difference/magnitude
                # moving representative closer
                self.representatives[i] = representative + ((fraction*magnitude) * direction)

    
# our CURE model
class CURE_Algorithm():

    # k is number of clusters
    # p is the fraction to move representatives closer to the centroid for each cluster
    # i.e if you put p=0.1, representatives will move 10% closer to the centroid
    # r is the number of representatives to choose for each cluster
    def __init__(self,k,p,r):
        self.k = k
        self.p = p
        self.r = r
        self.clusters =[]
        

    
    # sample is some chosen examples that the user decides to use to initialize the CURE algorithm
    def process_initial(self,sample):
        # lets just use sklearn agglomerative clustering to generate a hierarchical clustering
        # ward linkage minimizes variance of combined clusters
        model = AgglomerativeClustering(n_clusters=self.k,affinity='euclidean',linkage='ward').fit(sample)
        # creating our cure_cluster object based on the clustering received 

        #predictions are label where the predictions[i] = cluster index that sample i belongs to
        predictions = model.labels_
        
        # creating a mapping {clusterIndex: [examples assigned to this cluster]}
        cluster_example_mapping = {}
        for i,prediction in enumerate(predictions):
            if prediction not in cluster_example_mapping:
                cluster_example_mapping[prediction] = [sample[i]]
            else:
                cluster_example_mapping[prediction].append(sample[i])
        
        #computing cluster centroids and assigning representative points
        for cluster_number in cluster_example_mapping:
            centroid = np.sum(cluster_example_mapping[cluster_number])/len(cluster_example_mapping[cluster_number])
            cluster_object = CURE_Cluster(centroid,self.r)
            cluster_object.set_representatives(cluster_example_mapping[cluster_number])
            self.clusters.append(cluster_object)

        # for each cluster, moving representatives p closer to the centroid
        for cluster in self.clusters:
            cluster.move_representatives(self.p)

        # now, we have our clusters, we can tag outside of this model or call the tagging function individually afterwards...


    # tagging function that just returns the cluster index that this example is closest to
    # find the representative point closest to p, and assign p to that cluster
    def tag(self,example):
        distance_cluster = None
        for i,cluster in enumerate(self.clusters):
            # find the closest representative point
            for representative in cluster.representatives:
                distance = np.linalg.norm(example-representative)
                if distance_cluster is None or distance_cluster[0] > distance:
                    distance_cluster = (distance,i)
        # return the cluster of the representative point that this example is closest to
        return distance_cluster[1]


    
    
