from sklearn.cluster import KMeans, DBSCAN, OPTICS

def DBSCAN_labels(data, eps=0.5):
    """Divide data via DBSCAN
    """
    results = DBSCAN(eps=eps).fit(data)
    return results.labels_

def kmeans(data, clusters):
    """Divide data via K-means 
    """
    labels = KMeans(n_clusters=clusters, random_state=0).fit_predict(data)
    return labels

def optics(data, min_points):
    """Divide data via OPTICS
    """
    labels = OPTICS(min_points).fit_predict(data)
    return labels