import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd

rndn3d = np.random.randn(500, 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rndn3d[:, 0], rndn3d[:, 1], rndn3d[:, 2])
plt.show()

pca = PCA(n_components=3)
pca.fit(rndn3d)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

rndn3d1 = np.random.randn(500, 3)
pca.fit(rndn3d1)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

## on retrouve les meme valeurs ce qui coherent car les data provenant de la meme distribution, on s'attend à obtenir les memes projections


s1 = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 0.2]])
r1 = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])
rndef = rndn3d.dot(s1).dot(r1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rndef[:, 0], rndef[:, 1], rndef[:, 2])
plt.show()
PCA2 = PCA(n_components=3)
PCA2.fit(rndef)
print(PCA2.explained_variance_)
print(PCA2.explained_variance_ratio_)


s1 = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 0.2]])
r1 = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])
rndef = rndn3d1.dot(s1).dot(r1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rndef[:, 0], rndef[:, 1], rndef[:, 2])
plt.show()
PCA2 = PCA(n_components=3)
PCA2.fit(rndef)
print(PCA2.explained_variance_)
print(PCA2.explained_variance_ratio_)


### on a un axe principal seulement  88% , cela provient de l'allure des points ils sont situés sur un plan et de maniere symétrique le long d'une droite contenue dans ce plan
### on retrouve les valeurs des ratios des valeurs propres
### idem si l'on regere une loi normale on obtient les memes ordres de grandeurs


url = 'http://cedric.cnam.fr/~crucianm/src/mammals.csv'
mammals = pd.read_csv(url, delimiter=';', skiprows=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
noms = pd.read_csv(url, delimiter=';', skiprows=1, usecols=[0])
pcaM = PCA()
pcaM.fit(mammals)
print(pcaM.explained_variance_ratio_)
print(pcaM.explained_variance_)

##only one direction since the value are not centered reduced one componnent has huge eigen value

# we center the value
mammalsCR = preprocessing.scale(mammals)
pcaCR = PCA()
pcaCR.fit(mammalsCR)
print(pcaCR.explained_variance_ratio_)
print(pcaCR.explained_variance_)
plt.bar(range(len(pcaCR.explained_variance_)), height=pcaCR.explained_variance_)
plt.show()

##en centrant les valeurs on obtient des ratio mieux distribués et une décroissance dplus lisse

mNt = pcaCR.transform(mammalsCR)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(noms)):
    x, y, z = mNt[i, 0], mNt[i, 1], mNt[i, 2]
    ax.scatter(x, y, z)
    ax.text(x, y, z, str(noms.iloc[i]))
plt.show()

###leaf
url = 'http://cedric.cnam.fr/~crucianm/src/leaf.csv'
leafs = pd.read_csv(url, delimiter=',')
leafN = preprocessing.scale(leafs.iloc[:, 2:])
pca = PCA()
pca.fit(leafN)
plt.plot(pca.explained_variance_ratio_, '.')
plt.show()
mNt = pca.transform(leafN)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(leafs.iloc[:,1])):
    x, y, z = mNt[i, 0], mNt[i, 1], mNt[i, 2]
    ax.scatter(x, y, z)
    ax.text(x, y, z, str(leafs.iloc[i,1]))
plt.show()
