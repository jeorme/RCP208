import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# générer l'échantillon à partir de deux lois normales
N = 100000
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                        np.random.normal(5, 1, int(0.4 * N)),np.random.normal(8, 1, int(0.3 * N))))[:, np.newaxis]

# préparer les points où on calculera la densité
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

# préparation de l'affichage de la vraie densité, qui est celle à partir
#  de laquelle les données ont été générées (voir plus haut)
# la pondération des lois dans la somme est la pondération des lois
#  dans l'échantillon généré (voir plus haut)
true_density = (0.3*norm(0,1).pdf(X_plot[:,0]) + 0.4*norm(5,1).pdf(X_plot[:,0])+0.3*norm(8,1).pdf(X_plot[:,0]))

# estimation de densité par noyaux gaussiens
kde = KernelDensity(kernel='gaussian', bandwidth=.75).fit(X)

# calcul de la densité pour les données de X_plot
density = np.exp(kde.score_samples(X_plot))

# affichage : vraie densité et estimation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill(X_plot[:,0], true_density, fc='b', alpha=0.2, label='Vraie densité')
ax.plot(X_plot[:,0], density, '-', label="Estimation")
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
ax.legend(loc='upper left')
plt.show()

Xg = kde.sample(N)
kde2 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(Xg)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill(X_plot[:,0], true_density, fc='b', alpha=0.2, label='Vraie densité')
ax.plot(X_plot[:,0], density, '-', label="Estimation")
ax.plot(X_plot[:,0], np.exp(kde2.score_samples(X_plot)), 'r-', label="Estimation2")
ax.legend(loc='upper left')
plt.show()

# générer l'échantillon
N = 20
kd = np.random.rand(N, 2)

# définir la grille pour la visualisation
grid_size = 100
Gx = np.arange(0, 1, 1/grid_size)
Gy = np.arange(0, 1, 1/grid_size)
Gx, Gy = np.meshgrid(Gx, Gy)

# définir la largeur de bande pour le noyau
bw = 0.05

# estimation, puis calcul densité sur la grille
kde3 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(kd)
Z = np.exp(kde3.score_samples(np.hstack(((Gx.reshape(grid_size*grid_size))[:,np.newaxis],
        (Gy.reshape(grid_size*grid_size)[:,np.newaxis])))))

# affichage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Gx, Gy, Z.reshape(grid_size,grid_size), rstride=1,
                    cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.scatter(kd[:,0], kd[:,1], -10)
plt.show()

# générer l'échantillon
N = 100
kd1 = np.random.randn(int(N/2), 2) + (1.0, 1.0)
kd2 = np.random.randn(int(N/2), 2)
kd = np.concatenate((kd1, kd2))

# définir la grille pour la visualisation
grid_size = 100
Gx = np.arange(-1, 3, 4/grid_size)
Gy = np.arange(-1, 3, 4/grid_size)
Gx, Gy = np.meshgrid(Gx, Gy)

# définir la largeur de bande pour le noyau
bw = 0.2

# estimation, puis calcul densité sur la grille
kde4 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(kd)
Z = np.exp(kde4.score_samples(np.hstack(((Gx.reshape(grid_size *
                                                         grid_size))[:,np.newaxis],
               (Gy.reshape(grid_size*grid_size)[:,np.newaxis])))))

# affichage
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Gx, Gy, Z.reshape(grid_size,grid_size), rstride=1,
                    cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.scatter(kd[:,0], kd[:,1], -10)
plt.show()