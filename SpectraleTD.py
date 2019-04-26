import numpy as np    # si pas encore fait
from numpy import newaxis
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

data1 = np.hstack((np.random.randn(100,3) + [3,3,3],(np.ones(100))[:,newaxis]))
data2 = np.hstack((np.random.randn(100,3) + [-3,-3,-3],(2 * np.ones(100))[:,newaxis]))
data3 = np.hstack((np.random.randn(100,3) + [-3,3,3],(3 * np.ones(100))[:,newaxis]))
data4 = np.hstack((np.random.randn(100,3) + [-3,-3,3],(4 * np.ones(100))[:,newaxis]))
data5 = np.hstack((np.random.randn(100,3) + [3,3,-3],(5 * np.ones(100))[:,newaxis]))
data = np.concatenate((data1,data2,data3,data4,data5))
print(data.shape)   # vérification

np.random.shuffle(data)
val=[]
for i in range(1,21):
    spectral = SpectralClustering(n_clusters=5,n_neighbors=i, eigen_solver='arpack', affinity='nearest_neighbors').fit(data[:,:3])
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(data[:,0], data[:,1], data[:,2], c=spectral.labels_)
    #plt.show()
    val.append(metrics.adjusted_rand_score(spectral.labels_, data[:,3]))
print(val)
plt.scatter(range(1,len(val)+1),val)
plt.show()
spectral = SpectralClustering(n_clusters=5, eigen_solver='arpack',
               affinity='rbf', gamma=1.0).fit(data[:,:3])
print(metrics.adjusted_rand_score(spectral.labels_, data[:,3]))