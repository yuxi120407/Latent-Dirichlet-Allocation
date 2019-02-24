# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:13:31 2019

@author: yuxi
"""

from scipy.stats import beta
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import numpy as np
#%%
a=2
b=3
x=np.arange(0.01,1,0.01)
y=beta.pdf(x,a,b)
plt.plot(x,y)
plt.title('Beta')
plt.xlabel('x')
plt.ylabel('density')
plt.show()
#%%
import matplotlib.tri as tri
corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=2)

plt.figure(figsize=(8, 4))
for (i, mesh) in enumerate((triangle, trimesh)):
    plt.subplot(1, 2, i+ 1)
    plt.triplot(mesh)
    plt.axis('off')
    plt.axis('equal')
    
#%%
# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
             for i in range(3)]
def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)

#%%
import functools 
class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     functools.reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * functools.reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])
        
        
        
        
#%% 
def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):
    #import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')        
        
        
#%%   
from matplotlib import cm     
draw_pdf_contours(Dirichlet([1, 2, 3],cmap=cm.coolwarm))        
        
        
#%%
#from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
alpha = np.array([1, 2, 3])
def fun(x, y):
    quantiles = np.array([x, y, 1-x-y])
    z = dirichlet.pdf(quantiles, alpha)
    return z  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0.01, 0.5, 0.01)
X, Y = np.meshgrid(x, y)     
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape) 
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)             
        
        
#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
from math import gamma
from operator import mul
from functools import reduce
sns.set(style='white', font_scale=1.2, font='consolas')

def plot_mesh(corners):
    """Subdivide the triangle into a triangular mesh and plot the original and subdivided triangles."""
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)
    
    plt.figure(figsize=(6, 4))
    for i, mesh in enumerate((triangle, trimesh)):
        plt.subplot(1, 2, i+1)
        plt.triplot(mesh)
        plt.axis('off')
        plt.axis('equal')    
        
        
class Dirichlet:
    """Define the Dirichlet distribution with vector parameter alpha."""
    def __init__(self, alpha):
        
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])
    
    def pdf(self, x):
        """Returns pdf value for `x`. """
        return self._coef * reduce(mul, [xx ** (aa-1) for (xx, aa) in zip(x, self._alpha)])
        
        
class PlotDirichlet:
    """
    Plot the Dirichlet distribution as a contour plot on a 2-Simplex.
    """
    def __init__(self, corners):
        self._corners = corners
        self._triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        # Midpoints of triangle sides opposite of each corner
        self._midpoints = [(corners[(i+1) % 3] + corners[(i+2) % 3]) / 2.0 for i in range(3)]
        
    def xy2bc(self, xy, tol=1.e-3):
        """Map the x-y coordinates of the mesh vertices to the simplex coordinate space (aka barycentric coordinates).
        Here we use a simple method that uses vector algebra. For some values of alpha, calculation of the Dirichlet pdf 
        can become numerically unstable at the boundaries of the simplex so our conversion function will take an optional 
        tolerance that will avoid barycentric coordinate values directly on the simplex boundary.        
        """
        s = [(self._corners[i] - self._midpoints[i]).dot(xy - self._midpoints[i]) / 0.75 for i in range(3)]
        return np.clip(s, tol, 1.0-tol)
        
    def draw_pdf_contours(self, ax, dist, label=None, nlevels=200, subdiv=8, **kwargs):
        """Draw pdf contours for a Dirichlet distribution"""
        # Subdivide the triangle into a triangular mesh
        refiner = tri.UniformTriRefiner(self._triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)
        
        # convert to barycentric coordinates and compute probabilities of the given distribution 
        pvals = [dist.pdf(self.xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    
        ax.tricontourf(trimesh, pvals, nlevels, **kwargs)
        #plt.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75**0.5)
        ax.set_title(str(label))
        ax.axis('off') 
        return ax

        
if __name__ == '__main__':
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    plot_dirichlet = PlotDirichlet(corners)
   
    f, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0, 0]
    alpha = (0.85, 0.85, 0.85)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[0, 1]
    alpha = (1, 1, 1)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[0, 2]
    alpha = (5, 5, 5)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[1, 0]
    alpha = (1, 2, 3)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[1, 1]
    alpha = (2, 5, 10)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[1, 2]
    alpha = (50, 50, 50)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)


    f.savefig('../figures/dirichlet.png', bbox_inches='tight', transparent=True)
        
        
        
        
        
        
        
        
        
        
        