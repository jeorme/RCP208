import numpy as np    # si pas encore fait
from numpy import newaxis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

d1 = np.random.uniform(0,1,(100,3)) + [3,3,3]  # génération 100 points 3D suivant loi uniforme
d2 = np.random.uniform(0,1,(100,3))  + [-3,-3,-3] # génération 100 points 3D suivant loi uniforme
d3 = np.random.uniform(0,1,(100,3)) + [-3,3,3]  # génération 100 points 3D suivant loi uniforme
d4 = np.random.uniform(0,1,(100,3))  + [-3,-3,3] # génération 100 points 3D suivant loi uniforme
d5 = np.random.uniform(0,1,(100,3))  + [3,3,-3] # génération 100 points 3D suivant loi uniforme

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
print(metrics.jaccard_similarity_score(kmeans.labels_, data[:,3]))

rand= []
jacard=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=5, n_init=1, init='k-means++').fit(data[:, :3])
    rand.append(metrics.adjusted_rand_score(kmeans.labels_, data[:,3]))
    jacard.append(metrics.jaccard_similarity_score(kmeans.labels_, data[:,3]))
print(np.mean(rand),np.std(rand))
print(np.mean(jacard),np.std(jacard))

##texture

textures = np.loadtxt('texture.dat')
np.random.shuffle(textures)
kmeans = KMeans(n_clusters=11).fit(textures[:,:40])
print(metrics.adjusted_rand_score(kmeans.labels_, textures[:,40]))

lda = LinearDiscriminantAnalysis()
lda.fit(textures[:,:40],textures[:,40])
texturest = lda.transform(textures[:,:40])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(texturest[:,0], texturest[:,1], texturest[:,2], c=textures[:,40])
plt.show()
print(lda.score(textures[:,:40],textures[:,40]))
print(lda.explained_variance_ratio_)
plt.bar(range(len(lda.explained_variance_ratio_)),lda.explained_variance_ratio_)
plt.show()
kmeans = KMeans(n_clusters=11).fit(texturest[:,:10])
print(metrics.adjusted_rand_score(kmeans.labels_, textures[:,40]))

##test various component
scores=[]
for i in range(1,21):
    lda = LinearDiscriminantAnalysis(n_components=i)
    lda.fit(textures[:, :40], textures[:, 40])
    texturest = lda.transform(textures[:, :40])
    print(lda.score(textures[:, :40], textures[:, 40]))
    kmeans = KMeans(n_clusters=11).fit(texturest[:, :10])
    print(metrics.adjusted_rand_score(kmeans.labels_, textures[:, 40]))
    scores.append(metrics.adjusted_rand_score(kmeans.labels_, textures[:, 40]))

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
kmeans = KMeans(n_clusters=11).fit(texturest[:, :10])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(texturest[:,0], texturest[:,1], texturest[:,2], c=kmeans.labels_[:])
plt.show()
