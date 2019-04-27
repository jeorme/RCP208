import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# # générer l'échantillon
# N = 100
# X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
#            np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
#
# # préparer les données où on calculera la densité
# X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
#
# true_density = (0.3*norm(0,1).pdf(X_plot[:,0]) + 0.7*norm(5,1).pdf(X_plot[:,0]))
#
# # estimation par mélange gaussien, avec le « bon » nombre de composantes
# gmm = GaussianMixture(n_components=2,n_init=3).fit(X)
#
# if gmm.converged_:
#     print(gmm.lower_bound_)
#
#
# # calcul de la densité pour les données de X_plot
# density = np.exp(gmm.score_samples(X_plot))
#
# # affichage : vraie densité et estimation
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.fill(X_plot[:,0], true_density, fc='b', alpha=0.2, label='Vraie densité')
# ax.plot(X_plot[:,0], density, '-', label="Estimation")
# ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
# ax.legend(loc='upper left')
# plt.show()
# cv = []
# lg=[]
# for i in range(1,20):
#     gmm = GaussianMixture(n_components=i, n_init=3).fit(X)
#     lg.append(gmm.lower_bound_)
#
# plt.scatter(range(1,len(lg)+1),lg)
# plt.show()
#
# gmm = GaussianMixture(n_components=8, n_init=3).fit(X)
# # calcul de la densité pour les données de X_plot
# density = np.exp(gmm.score_samples(X_plot))
#
# # affichage : vraie densité et estimation
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.fill(X_plot[:,0], true_density, fc='b', alpha=0.2, label='Vraie densité')
# ax.plot(X_plot[:,0], density, '-', label="Estimation")
# ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
# ax.legend(loc='upper left')
# plt.show()
#
#
# ###multi dim :
#
# md1 = 1.5 * np.random.randn(200,2) + [3,3]
# md2 = np.random.randn(100,2).dot([[2, 0],[0, 0.8]]) + [-3, 3]
# md3 = np.random.randn(100,2) + [3, -3]
# md = np.concatenate((md1, md2, md3))
#
# # préparer les données où on calculera la densité
# grid_size = 100
# Gx = np.arange(-10, 10, 20/grid_size)
# Gx.shape
# (100,)
# Gy = np.arange(-10, 10, 20/grid_size)
# Gx, Gy = np.meshgrid(Gx, Gy)
# Gx.shape
# (100, 100)
#
# # estimation par mélange gaussien
# gmm = GaussianMixture(n_components=3,n_init=3).fit(md)
# if gmm.converged_:
#     print(gmm.lower_bound_)
#
#
# # calcul de la densité pour les données de la grille
# density = np.exp(gmm.score_samples(np.hstack(((Gx.reshape(grid_size*grid_size))[:,np.newaxis],
#         (Gy.reshape(grid_size*grid_size)[:,np.newaxis])))))
#
# # affichage : données et estimation
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Gx, Gy, density.reshape(grid_size,grid_size), rstride=1,
#                     cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
# ax.scatter(md[:,0], md[:,1], -0.025)
# plt.show()
#
# #####
# n_max = 8    # nombre de valeurs pour n_components
# n_components_range = np.arange(n_max)+1
# aic = []
# bic = []
#
# # construction des modèles et calcul des critères
# for n_comp in n_components_range:
#     gmm = GaussianMixture(n_components=n_comp,covariance_type ='diag').fit(md)
#     aic.append(gmm.aic(md))
#     bic.append(gmm.bic(md))
#
# # normalisation des résultats obtenus pour les critères
# raic = aic/np.max(aic)
# rbic = bic/np.max(bic)
# print(raic)
# print(rbic)
# # affichage sous forme de barres
# xpos = np.arange(n_max)+1  # localisation des barres
# largeur = 0.3              # largeur des barres
# fig = plt.figure()
# plt.ylim([min(np.concatenate((rbic,raic)))-0.1, 1.1])
# plt.xlabel('Nombre de composantes')
# plt.ylabel('Score')
# plt.bar(xpos, raic, largeur, color='r', label="AIC")
# plt.bar(xpos+largeur, rbic, largeur, color='b', label="BIC")
# plt.legend(loc='upper left')
# plt.show()
#
### TEXTURE
# lecture des données et aplication de l'ACP
from sklearn.decomposition import PCA
textures = np.loadtxt('texture.dat')
pca = PCA().fit(textures[:,:40])
texturesp = pca.transform(textures[:,:40])

# construction du modèle de mélange, vérifications
gmm = GaussianMixture(n_components=11,n_init=3).fit(texturesp[:,:2])
if gmm.converged_:
    print(gmm.n_iter_)
    print(gmm.lower_bound_)


# préparer les données où on calculera la densité
grid_size = 100
xmin = 1.3*np.min(texturesp[:,0])
xmax = 1.3*np.max(texturesp[:,0])
Gx = np.arange(xmin, xmax, (xmax-xmin)/grid_size)
ymin = 1.3*np.min(texturesp[:,1])
ymax = 1.3*np.max(texturesp[:,1])
Gy = np.arange(ymin, ymax, (ymax-ymin)/grid_size)
Gx, Gy = np.meshgrid(Gx, Gy)

# calcul de la densité pour les données de la grille
density = np.exp(gmm.score_samples(np.hstack(((Gx.reshape(grid_size*grid_size))[:,np.newaxis],
        (Gy.reshape(grid_size*grid_size)[:,np.newaxis])))))

# affichage des résultats
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Gx, Gy, density.reshape(grid_size,grid_size), rstride=1,
                    cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.scatter(texturesp[:,0], texturesp[:,1], -0.25)
plt.show()

##comparaison avec textures sur les 2 premieres composantes
gmm2 = GaussianMixture(n_components=11,n_init=3).fit(textures[:,:2])
if gmm2.converged_:
    print(gmm2.n_iter_)
    print(gmm2.lower_bound_)

# calcul de la densité pour les données de la grille
density2 = np.exp(gmm2.score_samples(np.hstack(((Gx.reshape(grid_size*grid_size))[:,np.newaxis],
        (Gy.reshape(grid_size*grid_size)[:,np.newaxis])))))

# affichage des résultats
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Gx, Gy, density2.reshape(grid_size,grid_size), rstride=1,
                    cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.scatter(textures[:,0], textures[:,1], -0.25)
plt.show()

## determination du nombres de composantes
n_max = 8    # nombre de valeurs pour n_components
n_components_range = np.arange(n_max)+6
print(n_components_range)
aic = []
bic = []

# construction des modèles et calcul des critères
for n_comp in n_components_range:
        gmm = GaussianMixture(n_components=n_comp).fit(texturesp[:,:2])
        aic.append(gmm.aic(texturesp[:,:2]))
        bic.append(gmm.bic(texturesp[:,:2]))


# normalisation des résultats obtenus pour les critères
raic = aic/np.max(aic)
rbic = bic/np.max(bic)

# affichage sous forme de barres
xpos = n_components_range  # localisation des barres
largeur = 0.3              # largeur des barres
fig = plt.figure()
plt.ylim([min(np.concatenate((rbic,raic)))-0.02, 1.01])
plt.bar(xpos, raic, largeur, color='r', label="AIC")
plt.bar(xpos+largeur, rbic, largeur, color='b', label="BIC")
plt.legend(loc='upper left')
plt.show()

##projection sur les 2 premiers axes discriminant
lda = LinearDiscriminantAnalysis(n_components=2).fit(textures[:,:40],textures[:,40])
texture2 = lda.transform(textures[:,:40])
aic2=[]
bic2=[]
print(texture2.shape)
for n_comp in n_components_range:
        gmm = GaussianMixture(n_components=n_comp).fit(texture2[:,:])
        aic2.append(gmm.aic(texture2[:,:]))
        bic2.append(gmm.bic(texture2[:,:]))


# normalisation des résultats obtenus pour les critères
raic = aic2/np.max(aic2)
rbic = bic2/np.max(bic2)

# affichage sous forme de barres
xpos = n_components_range  # localisation des barres
largeur = 0.3              # largeur des barres
fig = plt.figure()
plt.ylim([min(np.concatenate((rbic,raic)))-0.02, 1.01])
plt.bar(xpos, raic, largeur, color='r', label="AIC")
plt.bar(xpos+largeur, rbic, largeur, color='b', label="BIC")
plt.legend(loc='upper left')
plt.show()

print(raic)
print(rbic)