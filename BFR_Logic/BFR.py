# this file holds the logic for the bfr algorithm


# the idea is that through 1 pass through the data, we can create clusters
# each cluster is represented by: number of examples, sum of examples, and sum squared of examples

# we have a discard set ( points close enough to a cluster to be summarized by it)
# we have a compression set (points closer to each other than to a cluster)
# (compression set is summarized but not assigned to a cluster)
# retained set (isolated points)

# initialization step: take some sample of the dataset and run standard k-means on it


import numpy as np
from sklearn.cluster import KMeans

# need to define mahanalobis distance as a distance function based on the strong assumption of BFR
def mahanalobis_distance(bfr_cluster, example):
    # getting stddev of cluster firstly
    stddev = bfr_cluster.compute_standard_deviation()
    
    # stddev may be zero due to floating point issues, so just add a small epsilon
    epsilon = 0.000001
    
    # normalizing the example
    normalized = (example-bfr_cluster.centroid)/(stddev+epsilon)

    # taking sqrt of sum of squares of the normalized features
    return np.sqrt(np.sum(np.square(normalized)))

class BFR_Cluster():
    
    # clusters have a centroid

    # clusters have number of examples

    # clusters have sum of vectors of example features

    # clusters have sum of vectors of squared example features (to easily get stddev)


    # we initialize this cluster with a given state
    def __init__(self, centroid, num_examples, sum_examples, sum_squared_examples):
        self.centroid = centroid
        self.num_examples = num_examples
        self.sum_examples = sum_examples
        self.sum_squared_examples = sum_squared_examples

    # adding examples to the cluster after another batch of processing
    def add_examples(self,add_num_examples, add_sum_examples, add_sum_squared_examples):
        # updating num examples
        self.num_examples += add_num_examples
        
        # stability check here
        # basically, if the squared value includes a zero, but the non-squared value does not, we are losing precision
        # so, in that case, we truncate the non-squared value to zero to try and counter eventual error buildup resulting in negative variance
        # replacing entry with zero if needed
        for j in range(len(add_sum_squared_examples)):
            if add_sum_squared_examples[j] ==0 and add_sum_examples[j]!=0:
                add_sum_examples[j]=0
                    
                
        

        # updating the sum and sum_squared
        self.sum_examples = np.add(self.sum_examples,add_sum_examples)
        self.sum_squared_examples = np.add(self.sum_squared_examples,add_sum_squared_examples)


    # need to recompute centroid after every set of updates
    def recompute_centroid(self):
        self.centroid = self.sum_examples / self.num_examples

    # helper function to compute standard deviation
    def compute_standard_deviation(self):
        if self.num_examples == 0:
            print('ERROR: ' + str(self.num_examples))
        variance = (self.sum_squared_examples / self.num_examples) - np.square((self.sum_examples/self.num_examples))
        
        #floating point error can lead to zero elements, so just take max with zero
        for i in range(len(variance)):
            variance[i]=max(0,variance[i])
        
        return np.sqrt(variance)

    # helper function to merge another cluster into this one
    def merge(self,other_cluster):
        self.num_examples += other_cluster.num_examples
        self.sum_examples = np.add(self.sum_examples,other_cluster.sum_examples)
        self.sum_squared_examples = np.add(self.sum_squared_examples,other_cluster.sum_squared_examples)

        # recomputing centroid
        self.recompute_centroid()

# our BFR Algorithm has clusters, a compressed set, and a retained set
class BFR_Algorithm():

    # a standard treshhold is sqrt(d) where d is the number of features
    def __init__(self,k,distance_threshold, compression_combination_threshold):
        # need to initialize some k and a distance threshold to determine if something is close or not to an existing cluster
        self.k = k
        self.distance_threshold = distance_threshold
        self.compression_combination_threshold = compression_combination_threshold
        # actual main bfr clusters
        self.clusters=[]
        # compressed clusters
        self.compressed=[]
        # retained or isolated points
        self.retained = []
        

    
    # sample is some chosen examples that the user decides to use to initialize the bfr algorithm
    def process_initial(self,sample):
        # in the first step, we can use some main memory algorithm to cluster a sample of points
        # lets just use k-means in sk learn library and then update our results
        model = KMeans(n_clusters=self.k).fit(sample)
        # creating our bfr_cluster object based on the clustering received 
        clusters = model.cluster_centers_
        predictions = model.labels_

        for cluster in clusters:
            self.clusters.append(BFR_Cluster(cluster,0,0,0))
        for i,prediction in enumerate(predictions):
            self.clusters[prediction].add_examples(1,sample[i],np.square(sample[i]))

    
    '''
        # in the first iteration we use some main memory algorithm to cluster, so we can just use k-means in spark api and feed this class the results
        # clusters is a list of clusters
        # example_cluster_pairs is a list of lists where the first index is the cluster index, and the inner list is the list of datapoints assigned to that cluster summed, and then the second list is the squared examples
        def process_initial(clusters,example_cluster_pairs): 
            for i,cluster in clusters.enumerate():
                examples = example_cluster_pairs[i]
                self.clusters.append(\
                            BFR_Cluster(cluster,len(examples[0]),examples[0],examples[1])\
                        )
    '''
    
    # given a batch, we need to update the clusters, compressed set, and retained set
    # we are given examples and their ids
    # we return a dictionary of id to example mappings (basically if a given id can be summarized, we return its cluster assignment)
    def process_batch(self,examples,ids):
        # we first assign any examples that can be summarized to clusters
        # then, we can use a main memory algorithm to cluster the remaining examples and summarize them
        id_mapping = []
        for i,example in enumerate(examples):
            added = False
            for j,bfr_cluster in enumerate(self.clusters):
                # testing if this example is close to the cluster or not
                if mahanalobis_distance(bfr_cluster,example) < self.distance_threshold:
                    # we add this point to the cluster
                    bfr_cluster.add_examples(1,example,np.square(example))
                    id_mapping.append((ids[i],j))
                    # dont want to add this point to multiple clusters
                    added = True
                    break
            if not added:
                # if this example was not added to a cluster, we should add it to remaining set
                # this will be represented as an (example, id) tuple
                self.retained.append((example,ids[i]))

        # running a main memory algorithm to cluster remaining examples 
        if len(self.retained)>0 and len(self.retained)>=self.k:
            model = KMeans(n_clusters=self.k).fit([retained[0] for retained in self.retained])

            # creating our bfr_cluster object based on the clustering received 
            clusters = model.cluster_centers_
            predictions = model.labels_

            # creating cluster objects
            cluster_objects = [BFR_Cluster(cluster,0,0,0) for cluster in clusters]
            for i,prediction in enumerate(predictions):
                cluster_objects[prediction].add_examples(1,self.retained[i][0],np.square(self.retained[i][0]))

            # adding clusters to our compressed set
            # for clusters in the compressed set, they are represented as (cluster,[ids]) to represent the ids that were assigned to that CS
            for cluster in cluster_objects:
                self.compressed.append((cluster,[]))

            further_isolated = []
            while len(self.retained)>0:
                tup = self.retained.pop()
                example = tup[0]
                id_val = tup[1]
                added = False
                for cs_cluster in self.compressed:
                    # testing if this example is close to the cluster or not
                    if mahanalobis_distance(cs_cluster[0],example) < self.distance_threshold:
                        # we add this point to the cluster
                        cs_cluster[0].add_examples(1,example,np.square(example))
                        # adding this particular id to the "memory"
                        cs_cluster[1].append(id_val)
                        # dont want to add this point to multiple clusters
                        added = True
                        break
                if not added:
                    # if this example was not added to a cluster, we should add it to remaining set
                    further_isolated.append(tup)

            # updating isolated points
            self.retained = further_isolated
            
                
        # recomputing centroids for cs and for main cluster set
        for cluster in self.compressed:
            cluster[0].recompute_centroid()

        for cluster in self.clusters:
            cluster.recompute_centroid()

        # finally, we can consider merging compressed sets
        self.merge_compressed()
        
        # returning the id mapping of examples which have been classified so far
        return id_mapping


    # deciding if we should merge any compressed sets together
    def merge_compressed(self):
        merges_done = True
        # loop breaks once we iterate through each cluster pair without a merge
        while merges_done:
            merges_done = False
            first_index = None
            second_index = None
            # to allow us to merge continuously as much as possible, we will just break whenever we merge
            for i,cluster in enumerate(self.compressed):
                for j in range(i+1,len(self.compressed)):
                    other_cluster = self.compressed[j]
                    # computing variances
                    first_variance = np.square(cluster[0].compute_standard_deviation())
                    second_variance = np.square(other_cluster[0].compute_standard_deviation())
                    if np.all(first_variance + second_variance < self.compression_combination_threshold):
                        # then we combine these compressed sets
                        merges_done = True
                        first_index = i
                        second_index = j
                        break
                if merges_done:
                    break
            if merges_done:
                # we need to merge index i and index j clusters
                # arbitrarily, we will merge j into i, and we will remove j from the list
                self.compressed[i][0].merge(self.compressed[j][0])
                # updating the ids that are now in this compressed cluster
                self.compressed[i][1].extend(self.compressed[j][1])
                # removing the other cluster, since not needed after the merge
                self.compressed.pop(j)


    # if we finished passing through data, we need to assign CS and retained points to their nearest cluster
    def finalize(self):
        id_mapping = []
        cs_cluster_pair = []
        for cs in self.compressed:
            nearest_cluster = None
            nearest_distance = None
            for i,cluster in enumerate(self.clusters):

                dist = mahanalobis_distance(cluster,cs[0].centroid)
                if nearest_distance is None or nearest_distance > dist:
                    nearest_cluster = cluster
                    nearest_distance = dist
                
            cs_cluster_pair.append((cs,i))
        for tup in self.retained:
            example = tup[0]
            id_val = tup[1]
            nearest_cluster = None
            nearest_cluster_index = None
            nearest_distance = None
            for i,cluster in enumerate(self.clusters):
                dist = mahanalobis_distance(cluster,example)
                if nearest_distance is None or nearest_distance > dist:
                    nearest_cluster = cluster
                    nearest_distance = dist
                    nearest_cluster_index = i
            # adding this point to the nearest cluster we found
            nearest_cluster.add_examples(1,example,np.square(example))
            id_mapping.append((id_val,nearest_cluster_index))

        # merging compressed sets now to their destination clusters
        for cs_index in cs_cluster_pair:
            self.clusters[cs_index[1]].merge(cs_index[0][0])
            # adding all the ids from this compressed set to the cluster
            for example_id in cs_index[0][1]:
                id_mapping.append((example_id,cs_index[1]))

        # clearing retained and cs sets
        self.compressed.clear()
        self.retained.clear()

        # recomputing centroids
        for cluster in self.clusters:
            cluster.recompute_centroid()
            
        # returning the final id_mapping
        return id_mapping

