#Name: David Li
#Email: sli857@wisc.edu
#NetID: sli857
#CS Login: sli857

import sys
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram
from copy import deepcopy

def load_data(filepath):
    data = []
    with open(filepath, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if (row[0] == ""):
                continue
            rowData = {
                "": row[0],
                "Country": row[1],
                "Population": row[2],
                "Net migration": row[3],
                "GDP ($ per capita)": row[4],
                "Literacy (%)": row[5],
                "Phones (per 1000)": row[6],
                "Infant mortality (per 1000 births)": row[7]
            }
            data.append(rowData)
    return data

def calc_features(row):
    features = np.zeros((6,), dtype="float64")
    features[0] = row["Population"]
    features[1] = row["Net migration"]
    features[2] = row["GDP ($ per capita)"]
    features[3] = row["Literacy (%)"]
    features[4] = row["Phones (per 1000)"]
    features[5] = row["Infant mortality (per 1000 births)"]
    return features


def hac(features):
    size = len(features)
    #features = normalize_features(features)
    count = 0
    clusters = {}
    process = np.empty(((size - 1), 4))
    for i in range(size):
        clusters[i] = [features[i]]
    while(count < size - 1):
        # print()
        min = sys.maxsize
        clust1 = 0
        clust2 = 0
        for key1 in clusters:
            # print(key1)
            for key2 in clusters:
                if(key1 >= key2): 
                    continue
                else:
                    # print(" ", key2)
                    #print(count, key1, key2, clusters[key1], clusters[key2])
                    curr = calc_completeLinkageDistance(clusters[key1], clusters[key2])
                    # if(curr == min):
                    #     print("EQUAL")
                    if(curr < min):
                        min = curr
                        clust1 = key1
                        clust2 = key2
        process[count] = [int(clust1), int(clust2), min, len(clusters[clust1]) + len(clusters[clust2])]
        clusters[count + size] = clusters[clust1] + clusters[clust2]
        clusters.pop(clust1)
        clusters.pop(clust2)
        count += 1
    return process

def calc_completeLinkageDistance(first, second):
    # first = np.array(first).reshape(len(first), 6)
    # second = np.array(second).reshape(len(second), 6)
    distances = distance_matrix(first, second, p=2)
    #distances = np.array(distances)
    max = 0
    for i in range(len(first)):
        for j in range((len(second))):
            if(distances[i, j] > max):
                max = distances[i, j]
    return max
     
def fig_hac(Z, names):
    # fig = plt.figure()
    # dn = dendrogram(Z, labels=names, leaf_rotation=90)
    # plt.tight_layout()
    # fig.draw()
    
    fig = plt.figure()
    dn1 = dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig


def normalize_features(features):
    means, stds = calc_meansNstd(features)
    #print(means, stds)
    copy = deepcopy(features)
    for row in copy:
        for i in range(6):
            row[i] = (row[i] - means[i]) / stds[i]
    return copy

# def calc_distance(first, second):
#     return math.sqrt(sum(math.pow(first[i] - second[i], 2) for i in range(6)))

def calc_meansNstd(dataset):
    size = len(dataset)
    means = [0 for i in range(6)]
    stds = [0 for i in range(6)]
    for row in dataset:
        for i in range(6):
            means[i] += row[i]/size

    for i in range(6):
        stds[i] = math.sqrt(sum(math.pow(x[i] - means[i], 2) for x in dataset) / size)
    #print( means, stds, sep = "\n")
    return means, stds


def main():
    data = load_data("./countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = len(features)
    #Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    #fig = fig_hac(Z_raw, country_names[:n])
    fig2 = fig_hac(Z_normalized, country_names[:n])
    plt.show()

if __name__ == '__main__':
    main()
