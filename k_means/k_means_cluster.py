import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, K):
    """
    Randomly initialize K cluster centroids from the data
    
    arguments:
    X -- our data that we want to cluster
    K -- number of clusters
    
    return:
    centroids -- initialized centroids
    """
    idx = np.random.choice(X.shape[0], K, replace = False)
    centroids = X[idx,:]
    return centroids
    

def compute_distance(X, K_clusters):
    """
    compute the distance between the examples of our data and the centroids of the clusters
    
    arguments:
    X -- our data
    K_clusters -- centroids of the K clusters
    
    return:
    dis -- the distance
    """
    dis = np.linalg.norm((X-K_clusters),2,axis=1)**2
    return dis
    
    
    
def k_means (X, K):
    """
    derive the K clusters from the data
    
    arguments:
    X -- our data
    K -- number of clusters
    
    return:
    groups -- the labels (clusters) assigned to the examples in the data
    K_clusters -- centers of clusters
    
    """
    K_clusters = initialize_centroids(X, K)
    m = X.shape[0]
    dif = 1
    while (dif > 10**(-7)): # we stop when the centroids almost don't move
        groups = np.empty(m)
        K_clusters_old = K_clusters
        #cluster assignment step
        for i in range(m):
            groups[i] = np.argmin(compute_distance(X[i,:],K_clusters))
        #centroids update step
        for k in range(K):
            K_clusters[k,:] = np.mean(X[groups==k,:],axis=0)
        dif = np.linalg.norm(K_clusters-K_clusters_old, 2) / (np.linalg.norm(K_clusters, 2) + np.linalg.norm(K_clusters_old, 2))
    return groups.astype(int), K_clusters
    
    
def compute_cost(X, groups, K_clusters):
    """
    compute the cost function (also called distortion of the training examples) that we want to minimize.
    It represents the average of the distances of every training example to its corresponding cluster centroid
    
    arguments:
    X -- the data
    groups -- labels of clusters assignedto each example of the data
    K_clusters -- centroids of the clusters
    
    return:
    cost -- the cost function to minimize
    """
    m = X.shape[0]
    dis = np.empty(m)
    for i in range(m):
        dis[i] = compute_distance(X[i,:].reshape(1,X.shape[1]), K_clusters[groups[i],:].reshape(1,X.shape[1]))
        cost = (1/m)*np.sum(dis)
    return cost
    
    
def k_means_iter(X, K, n_iter):
    """
    run the k_means algorithm on many different random initialization and then we keep the clustering that gace the lowest cost.
    
    arguments:
    X -- our data
    K -- number of clusters
    n_iter -- number of iterations
    
    return:
    cluster_groups -- the labels (clusters) assigned to the examples in the data
    cluster_centroids -- centers of clusters
    """
    cost=[]
    centroids_dict={}
    for i in range (n_iter):
        groups, K_clusters=k_means(X, K)
        cost.append(compute_cost(X, groups, K_clusters))
        centroids_dict['groups'+str(i)]=groups
        centroids_dict['K_clusters'+str(i)]=K_clusters
    opt_cost_index=cost.index(min(cost))
    cluster_groups=centroids_dict['groups'+str(opt_cost_index)]
    cluster_centroids=centroids_dict['K_clusters'+str(opt_cost_index)]
    return cluster_groups,cluster_centroids 
    
    
#def main():
    
    
    
#if __name__=="__main__":
 #   main()