import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# générer l'échantillon
md1 = 1.5 * np.random.randn(200,2) + [3,3]
md2 = np.random.randn(100,2).dot([[2, 0],[0, 0.8]]) + [-3, 3]
md3 = np.random.randn(100,2) + [3, -3]
md = np.concatenate((md1, md2, md3))

# préparer les données où on calculera la densité
grid_size = 100
Gx = np.arange(-10, 10, 20/grid_size)
print(Gx.shape)

Gy = np.arange(-10, 10, 20/grid_size)
Gx, Gy = np.meshgrid(Gx, Gy)
print(Gx.shape)


# estimation par mélange gaussien
gmm = GaussianMixture(n_components=3,n_init=3).fit(md)
if gmm.converged_:
    print(gmm.lower_bound_)


# calcul de la densité pour les données de la grille
density = np.exp(gmm.score_samples(np.hstack(((Gx.reshape(grid_size*grid_size))[:,np.newaxis],
        (Gy.reshape(grid_size*grid_size)[:,np.newaxis])))))

# affichage : données et estimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Gx, Gy, density.reshape(grid_size,grid_size), rstride=1,
                    cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.scatter(md[:,0], md[:,1], -0.025)
plt.show()