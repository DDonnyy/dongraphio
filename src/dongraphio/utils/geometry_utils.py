import numpy as np
import pandas as pd
from shapely import Point
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def clusterize_kmeans_geo_points(loc: pd.Series, noise_points: [Point]) -> pd.DataFrame:
    data = np.array([[p.x, p.y] for p in loc["geometry"]])
    noise = np.array([[p.x, p.y] for p in noise_points])
    data = np.append(data, noise, axis=0)

    best_silhouette = -1
    best_labels = None
    k_values = range(2, 10 if len(data) > 10 else 2)

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette = silhouette_score(data, labels)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_labels = labels

    data = data[: -len(noise)]
    if best_labels is not None:
        best_labels = best_labels[: -len(noise)]
    else:
        best_labels = [0 for _ in range(0, len(data))]

    data = pd.DataFrame({"label": best_labels, "geometry": [Point(p[0], p[1]) for p in data]})
    grouped_data = data.groupby("label")["geometry"].apply(list)
    grouped_data = pd.DataFrame(grouped_data).reset_index(drop=True)
    grouped_data.index = grouped_data.index + 1
    return grouped_data
