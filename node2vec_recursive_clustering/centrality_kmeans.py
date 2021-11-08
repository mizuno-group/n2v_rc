# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:05:49 2021
refer to https://hkawabata.github.io/technical-note/note/ML/k-means.html

--- scratch KMeans ---
determine initial centroid with network centrality
- Betweennes Centrality
- Degree Centrality



@author: I.Azuma
"""
import numpy as np
import networkx as nx
import itertools

class Scratch_KMeans():
    """
    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    labels_ : numpy array
        Labels of each point
    mu : numpy array
        centroid
    sse : float
        sum of squared errors in the cluster
    """
    
    def __init__(self, g, n_clusters, max_iter):
        self.g = g
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centrality = None
    
    def define_centrality(self,data:dict={}):
        """
        define centrality ranking
        Parameters
        ----------
        data : dict
            {'Gene1':0.003,'Gene2':0.05,'Gene3':0.0,...}

        Returns
        -------
        self.centrality : list
            [('Gene100',0.50),('Gene200',0.48),('Gene300':0.45),...]
            list contains tuple which has gene and its centrality value
        """
        if len(data)==0:
            #print('calc centrality')
            centrality = nx.betweenness_centrality(self.g) # betweenness centrality
        else:
            #print('use existing')
            centrality = data
        
        self.centrality = sorted(centrality.items(),key=lambda x : x[1],reverse=True)
    
    def fit(self, vec_df, spd_consideration=False):
        """
        perform KMenas clustering according to the network centrality information

        Parameters
        ----------
        vec_df : DataFrame
            Embedded vectors with node2vec. You can obtain this vectores by using 'vectorization.py'
        spd_consideration : bool
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        data = np.array(vec_df)
        n = len(data)
        dim = len(data[0])
        centrality_list = []
        for e in self.centrality:
            # the target gene is retained in the embedded vectors
            if e[0] in vec_df.index.tolist():
                centrality_list.append(e)
            else:
                pass
        # initialize the centroid
        self.init_centroid = self.__high_extraction(centrality=centrality_list,spd_consideration=spd_consideration)
        mu = np.array(vec_df.loc[self.init_centroid])
        for t in range(self.max_iter):
            cluster = np.zeros(n, dtype=int)
            cluster_size = np.zeros(self.n_clusters)
            cluster_sum = np.zeros([self.n_clusters, dim])
            for i in range(n):
                d_sq_min = np.inf
                for j in range(self.n_clusters):
                    d_sq = np.sum((data[i] - mu[j])**2) # Euclidean distance from each centroid
                    if d_sq < d_sq_min:
                        d_sq_min = d_sq
                        cluster[i] = j # attribute the target to the closest centroid at distance
                cluster_sum[cluster[i]] += data[i]
            mu_next = (cluster_sum.T / cluster_size).T # recalculate and update the centroid
            if np.all(mu == mu_next):
                break
            mu = mu_next
        self.labels_ = cluster
        self.mu = mu
        self.sse = self.__calc_sse(data)
    
    def __calc_sse(self, data):
        """
        calculate SSE
        """
        sse = 0
        for i in range(len(data)):
            sse += np.sum((data[i] - self.mu[self.labels_[i]])**2)
        return sse
    
    def __high_extraction(self,centrality:list,d_threshold:int=2,spd_consideration=False):
        """
        The shortest path distance (SPD) between candidate genes is also considered to determine the initial candidate values for centroid.
        If the SPD is not more than the threshold value, the candidate genes are not added.

        Parameters
        ----------
        centrality : list
            list contains tuple which has gene and its centrality value. e.g.[('Gene100',0.50),('Gene200',0.48),('Gene300':0.45),...]
        d_threshold : int
            The default is 2. Centroid candidates that are less than this value will not be added.
        spd_consideration : bool
            Variable whether to consider the SPD between candidates to determine the centroid. The default is False.

        Returns
        -------
        final_gene : list
            List of genes corresponding to the initia values of the KMeans clustering centroid

        """
        if spd_consideration:
            print("SPD consideration")
            count = 0
            final_gene = []
            for i in range(len(self.g.nodes())):
                t = [x[0] for x in centrality][i]
                if len(final_gene)==0: # no condtional branching in the first step
                    final_gene.append(t)
                    count += 1
                else:
                    target_gene = set(final_gene).union([t])
                    comb = list(itertools.combinations(target_gene, 2))
                    size_list = []
                    for c in comb:
                        # calculate the shoetest path length in each combination pair
                        test = nx.bidirectional_shortest_path(self.g, c[0], c[1]) #e.g. ("source","geneA","geneB","destination") --> SPD=3
                        size_list.append(len(test)-1)
                        if len(test)-1 <= d_threshold:
                            break # terminate
                        else:
                            pass
                    if min(size_list) > d_threshold: # meet the constraint and add the node as initial centroid position
                        final_gene.append(t)
                        count += 1
                    else:
                        #print(t," was denied")
                        pass
                if count == self.n_clusters: # obtain all the candidate numbers for initial centroid nodes
                    break
                else:
                    pass
        else:
            print("without SPD consideration")
            final_gene = [x[0] for x in centrality][0:self.n_clusters] # add them in order, starting from the head
        
        return final_gene

