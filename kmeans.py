from pyspark import SparkContext
import numpy as np
import sys
from math import radians, degrees, sin, cos, asin, acos

def euc_dis(x,y):
    return np.linalg.norm(np.array(x)-np.array(y))

def bc_dis(x,y):
    lon1, lat1, lon2, lat2 = map(radians, [x[0], x[1], y[0], y[1]])
    value = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)
    if value > 1:
        return 0
    return acos(value)

def addpoints(x,y):
    return x[0]+y[0], x[1]+y[1]

def closestpoint(x, centroid, f):
    min_dis = -1
    index = 0
    for i in range(len(centroid)):
        if min_dis == -1:
            min_dis = f(x,centroid[i])
        else:
            dis = f(x,centroid[i])
            if min_dis > dis:
                min_dis = dis
                index = i
    
    return index

def converge(old_cent, new_cent, f):
    means = 0;
    for i in range(len(old_cent)):
        means = means + f(old_cent[i],new_cent[i])
    return means

def computeCentroid(assignment,k):
    # count number of points under each index
    result = assignment.reduceByKey(lambda x,y: (addpoints(x[0],y[0]),x[1]+y[1])).map(lambda x: (x[0],(x[1][0][0]/x[1][1],x[1][0][1]/x[1][1]))).collect()
    new_cent = [(0,0)]*k
    for i in range(len(result)):
        new_cent[result[i][0]]=result[i][1]
    return new_cent

#def combineListsAsRDD(centroid,cluster):


def kmeans(data,k,f,convergeDist):
    centroid = data.takeSample(False, k, 2019)
    current_convergence = 1
    old_means = 0
    flag = 0
    while current_convergence > convergeDist:
        assignment = data.map(lambda x: (closestpoint(x,centroid,f),(x,1)))
        new_cent = computeCentroid(assignment,k)
        if flag == 0:
            old_means = converge(centroid,new_cent,f)
            flag = 1
            new_means = old_means
        elif old_means == 0:
            current_convergence = 0
        else:
            new_means = converge(centroid,new_cent,f)
            current_convergence = 1 - new_means/old_means
            old_means = new_means
        centroid = new_cent
    assignment = data.map(lambda x: (closestpoint(x,centroid,f),x)).groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
    return centroid, assignment

# input: data file path, k, distance function, convergeDist, save file path
def main():
    disfunc = ""
    sc = SparkContext()
    data = sc.textFile(sys.argv[1]).map(lambda x: x.split("\t")).filter(lambda x: len(x) != 1).map(lambda x: (float(x[0]), float(x[1]))).cache()
    if sys.argv[3] == "euc":
        disfunc = euc_dis
    elif sys.argv[3] == "bc":
        disfunc = bc_dis
    p, q = kmeans(data, int(sys.argv[2]), disfunc, float(sys.argv[4]))
    cat = []
    for c in q:
        cat.append((p[c[0]],c[1]))
    r = sc.parallelize(cat)
    r.coalesce(1).saveAsTextFile(sys.argv[5])

if __name__ == "__main__":
    main()


