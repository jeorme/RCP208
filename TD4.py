import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection  import train_test_split
import pandas as pd

s1 = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 0.01]])
r1 = np.array([[0.36,0.48,-0.8],[-0.8,0.6,0],[0.48,0.64,0.6]])
rndn3d1 = np.random.randn(500,3)
rndef1 = rndn3d1.dot(s1).dot(r1)
rndn3d2 = np.random.randn(500,3)
rndef2 = rndn3d2.dot(s1).dot(r1) + [0, 0, 1]
rndef = np.concatenate((rndef1, rndef2))
lcls1 = np.ones(500)
lcls2 = 2 * np.ones(500)
lcls = np.concatenate((lcls1, lcls2))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rndef[:, 0], rndef[:, 1], rndef[:, 2], c=lcls)
plt.show()

#discriminante analysis
lda = LinearDiscriminantAnalysis()
lda.fit(rndef,lcls)
rndt = lda.transform(rndef)
print(rndt.shape)
plt.plot(rndt, lcls, 'r+')
plt.show()

#reduction de la dimension puis AFD
pca = PCA(n_components=2)
pca.fit(rndef)
rndp = pca.transform(rndef)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(rndp[:,0], rndp[:,1], c=lcls)
plt.show()

lda = LinearDiscriminantAnalysis()
lda.fit(rndp,lcls)
rndpt = lda.transform(rndp)
plt.plot(rndpt, lcls, 'r+')
plt.show()

###texture
url = "http://cedric.cnam.fr/~crucianm/src/texture.dat"
textures = np.loadtxt(url)
pcaT = PCA(n_components=3)
pcaT.fit(textures[:,0:39])
print(pcaT.explained_variance_ratio_)
texturesPCA = pcaT.transform(textures[:,0:39])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(texturesPCA[:,0], texturesPCA[:,1],texturesPCA[:,2],c=textures[:,40])
plt.show()

#AFD
ldaT = LinearDiscriminantAnalysis(n_components=3)
ldaT.fit(textures[:,0:39],textures[:,40])
texturesAFD = ldaT.transform(textures[:,0:39])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(texturesAFD[:,0], texturesAFD[:,1],texturesAFD[:,2],c=textures[:,40])
plt.show()




train, test = train_test_split(textures, test_size=0.2)
ldaT = LinearDiscriminantAnalysis(n_components=3)
ldaT.fit(train[:,0:39],train[:,40])
trainAFD = ldaT.transform(train[:,0:39])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(trainAFD[:,0], trainAFD[:,1],trainAFD[:,2],c=train[:,40])
plt.show()

print(ldaT.score(train[:,0:39],train[:,40]))
print(ldaT.score(test[:,0:39],test[:,40]))

testAFD = ldaT.transform(test[:,0:39])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(testAFD[:,0], testAFD[:,1],testAFD[:,2],c=test[:,40])
plt.show()

#comparaison ACP // AFD pour les data leafs
url = 'http://cedric.cnam.fr/~crucianm/src/leaf.csv'
leafs = pd.read_csv(url, delimiter=',')
lda = LinearDiscriminantAnalysis()
lda.fit(leafs.iloc[:,2:],leafs.iloc[:,0])
leafl = lda.transform(leafs.iloc[:,2:])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(leafl[:,0],leafl[:,1],leafl[:,2],c=leafs.iloc[:,0])
plt.show()
##score AFD
print(lda.score(leafs.iloc[:,2:],leafs.iloc[:,0]))
#score PCA
leafN = preprocessing.scale(leafs.iloc[:, 2:])
pca = PCA()
pca.fit(leafN)
mNt = pca.transform(leafN)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mNt[:,0],mNt[:,1],mNt[:,2],c=leafs.iloc[:,0])
plt.show()


train, test = train_test_split(leafs, test_size=0.2)
ldatrain = LinearDiscriminantAnalysis()
ldatrain.fit(train.iloc[:,2:],train.iloc[:,0])
print(lda.score(train.iloc[:,2:],train.iloc[:,0]))
print(lda.score(test.iloc[:,2:],test.iloc[:,0]))