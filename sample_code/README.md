# sample code information

## 1. sample_code1.ipynb : How to handle 'vectorization.py'
- transform network structure to d-dimensional vectors using node2vec
1. load a network data
2. set parameter of node2vec
3. conduct node2vec and transform the network to d-dimensional vectors
4. obtain the d-dimensional vectors as csv format

## 2. sample_code2.ipynb : How to handle 'node2vec_recursive_clustering.py'
- perform unsupervised clustering to embedded vectors
1. load a network data and its embedded vectors in csv format (can be obtained with sample_code1.ipynb)
2. first step : determine the optimal number of clusters k using K-Means clustering with monitoring Q (modularity)
3. save the above first step result in Python pickle format
4. recursive step : perform the same operation recursively on the determined clusters
