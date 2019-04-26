import numpy as np    # si pas encore fait
from numpy import newaxis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import SpectralClustering

data1 = np.hstack((np.random.uniform(0,1,(100,3)) + [3,3,3],(np.ones(100))[:,newaxis]))
data2 = np.hstack((np.random.uniform(0,1,(100,3)) + [-3,-3,-3],(2 * np.ones(100))[:,newaxis]))
data3 = np.hstack((np.random.uniform(0,1,(100,3)) + [-3,3,3],(3 * np.ones(100))[:,newaxis]))
data4 = np.hstack((np.random.uniform(0,1,(100,3)) + [-3,-3,3],(4 * np.ones(100))[:,newaxis]))
data5 = np.hstack((np.random.uniform(0,1,(100,3)) + [3,3,-3],(5 * np.ones(100))[:,newaxis]))
data = np.concatenate((data1,data2,data3,data4,data5))
print(data.shape)   # vérification

np.random.shuffle(data)


##plot the point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3])
plt.show()
val=[]
for i in range(1,21):
    spectral = SpectralClustering(n_clusters=5, n_neighbors=i,eigen_solver='arpack', affinity='nearest_neighbors').fit(data[:,:3])
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(data[:,0], data[:,1], data[:,2], c=spectral.labels_)
    #plt.show()
    val.append(metrics.adjusted_rand_score(spectral.labels_, data[:,3]))
plt.scatter(range(1,len(val)+1),val)
plt.show()
val2=[]
for i in range(1,21):
    spectral = SpectralClustering(n_clusters=5, gamma=i, eigen_solver='arpack', affinity='rbf').fit(data[:,:3])
    val2.append(metrics.adjusted_rand_score(spectral.labels_, data[:,3]))

plt.scatter(range(1,len(val2)+1),val2)
plt.show()

##texture

textures = np.loadtxt('texture.dat')
print(textures.shape)
np.random.shuffle(textures)
spectral = SpectralClustering(n_clusters=11,eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=9).fit(textures[:,:40])
print(metrics.adjusted_rand_score(spectral.labels_, textures[:,40]))

##test various component
scores=[]
for i in range(1,21):
     lda = LinearDiscriminantAnalysis(n_components=i)
     lda.fit(textures[:, :40], textures[:, 40])
     texturest = lda.transform(textures[:, :40])
     print(lda.score(textures[:, :40], textures[:, 40]))
     spectral = SpectralClustering(n_clusters=11,eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=9).fit(texturest[:,:])
     print(metrics.adjusted_rand_score(spectral.labels_, textures[:, 40]))
     scores.append(metrics.adjusted_rand_score(spectral.labels_, textures[:, 40]))

plt.scatter(range(1,len(scores)+1),scores)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(textures[:,0], textures[:,1], textures[:,2], c=textures[:,40])
plt.show()
lda = LinearDiscriminantAnalysis(n_components=8)
lda.fit(textures[:, :40], textures[:, 40])
texturest = lda.transform(textures[:, :40])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(texturest[:,0], texturest[:,1], texturest[:,2], c=textures[:,40])
plt.show()
spectral = SpectralClustering(n_clusters=11,eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=9).fit(texturest[:,:])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(texturest[:,0], texturest[:,1], texturest[:,2], c=spectral.labels_[:])
plt.show()
print(metrics.adjusted_rand_score(spectral.labels_, textures[:, 40]))
