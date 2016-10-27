
import pandas as pd
import numpy as np
import random as rd
import math as m
from copy import deepcopy as dp


class Cluster:
    def __init__(self):
        self.new_centroid = np.array(0)
        self.old_centroid = np.array(0)
        self.elements = []
        self.backup = []
        self.label = ''
        self.b = 0
        self.m = 0
        self.error = 0

    def addelem(self, a):
        self.elements.append(a)

    def cleanup(self):
        self.backup = dp(self.elements)
        self.elements.clear()

    def calccentroid(self):
        self.new_centroid = 0
        for i in range(len(self.elements)):
            self.new_centroid += self.elements[i]
        self.new_centroid = self.new_centroid/len(self.elements)

    def isempty(self):
        if len(self.elements) == 0:
            return True
        else:
            return False

    def calclabel(self):
        self.b = 0
        self.m = 0
        for i in range(len(self.elements)):
            if self.elements[i][-1] == 2:
                self.b += 1
            else:
                self.m += 1
        if self.b > self.m:
            self.label = 'Benign'
        else:
            self.label = 'Malignant'

    def calcerror(self):
        if self.b > self.m:
            self.error = self.m/(self.b + self.m)
        else:
            self.error = self.b/(self.b + self.m)

def eucdist(a, b):
    s = 0
    # Excluding class variable(last column) from distance calculation
    for i in range(len(a)-1):
        s += pow((b[i] - a[i]), 2)
    di = m.sqrt(s)
    return di


def kmeans(d, k):
    clusters = [Cluster() for i in range(0, k)]
    threshold = 1
    t = 0
    # Initializing the clusters
    for l in range(len(clusters)):
        clusters[l].new_centroid = rd.choice(d)
    # Iterate until threshold is matched
    while threshold != 0:
        # Calcluating the distances to centroids
        for i in range(len(d)):
            mindist = []
            for j in range(len(clusters)):
                mindist.append(eucdist(clusters[j].new_centroid, d[i]))
            k1 = mindist.index(min(mindist))
            clusters[k1].addelem(d[i])
        threshold = 0
        # Recomputing the centroids
        for i in range(len(clusters)):
            if clusters[i].isempty():
                continue
            else:
                clusters[i].old_centroid = clusters[i].new_centroid
                clusters[i].calclabel()
                clusters[i].calcerror()
                clusters[i].calccentroid()
                clusters[i].cleanup()
                threshold += eucdist(clusters[i].old_centroid, clusters[i].new_centroid)
        t += 1
    return clusters, t


if __name__ == '__main__':
    df = pd.read_csv("DeltaClean.csv", sep=',', header=0)
    # Removing the SCN column while computing k-means
    df2 = df[df.columns[1:10]]
    errordict = {}
    for k in range(2, 6):
        totalerrors = []
        for j in range(20):
            totalerror = 0
            clusters1, t1 = kmeans(df2.values, k)
            for i in range(len(clusters1)):
                # print(len(clusters1[i].backup))
                totalerror += clusters1[i].error
            print("Converged in "+str(t1)+" iterations, with "+str(len(clusters1))+" clusters")
            print("Total error rate: " + str(totalerror))
            totalerrors.append(totalerror)
        errordict[k] = totalerrors
    print()
    print("Error rates for k = 2,3,4,5 run for 20 times each:")
    print(errordict)
