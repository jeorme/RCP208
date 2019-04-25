import numpy as np    # si pas encore fait
from numpy import newaxis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics

d1 = np.random.randn(100,3) + [3,3,3]  # génération 100 points 3D suivant loi normale centrée en [3,3,3]
d2 = np.random.randn(100,3) + [-3,-3,-3]
d3 = np.random.randn(100,3) + [-3,3,3]
d4 = np.random.randn(100,3) + [-3,-3,3]
d5 = np.random.randn(100,3) + [3,3,-3]
c1 = np.ones(100)      # génération des étiquettes du groupe 1
c2 = 2 * np.ones(100)  # génération des étiquettes du groupe 2
c3 = 3 * np.ones(100)
c4 = 4 * np.ones(100)
c5 = 5 * np.ones(100)
data1 = np.hstack((d1,c1[:,newaxis]))  # ajout des étiquettes comme 4ème colonne
data2 = np.hstack((d2,c2[:,newaxis]))
data3 = np.hstack((d3,c3[:,newaxis]))
data4 = np.hstack((d4,c4[:,newaxis]))
data5 = np.hstack((d5,c5[:,newaxis]))
data = np.concatenate((data1,data2,data3,data4,data5))  # concaténation des données dans une matrice
data.shape
(500, 4)
np.random.shuffle(data)

##plot the point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3])
plt.show()
kmeans = KMeans(n_clusters=5, n_init=1, init='k-means++').fit(data[:,:3])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=kmeans.labels_)
plt.show()
print(metrics.adjusted_rand_score(kmeans.labels_, data[:,3]))

kmeans = KMeans(n_clusters=5, n_init=10, init='random').fit(data[:,:3])
print(metrics.adjusted_rand_score(kmeans.labels_, data[:,3]))

##inertia
inertia = []
for nbC in range(1,20):
    kmeans = KMeans(n_clusters=nbC, n_init=10, init='k-means++').fit(data[:,:3])
    inertia.append(kmeans.inertia_)
