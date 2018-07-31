import numpy as np
import pdb
import matplotlib.pyplot as plt

from scipy.stats.distributions import norm
from sklearn.neighbors import KernelDensity

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GMM
class Cluster():

    """
    Class for different clustering methods

    Arguments:
        K       -- the number of clusters
        X       -- the data (default None);
        N       -- number of unique data points to generate (default: 0);
        plot    -- display output graphs

    """

    def __init__(self, K=None, X=None, N=10000, plot=True):
        """Initialisation"""
        self.K = K
        if X is None:
            if N == 0:
                raise Exception("If no data is provided, \
                        a parameter N (number of points) is needed")
            else:
                self.N = N
                self.X = self._init_gauss()

        self.plot = plot

    def _init_gauss(self):
        np.random.seed(0)
        x_grid = np.linspace(-5,5,self.N)
        x = np.concatenate([norm(-1,1).rvs(int(self.N*0.6)), norm(2,0.3).rvs(int(self.N*0.2))])
        pdf_true = (0.8*norm(-1,1).pdf(x_grid)+0.2*norm(2,0.3).pdf(x_grid))
        return x_grid,x,pdf_true

    def _create_blobs(self):
        X, y_true = make_blobs(n_samples=self.N, centers=self.K,
                cluster_std=0.5,random_state=0)
        #flip X for better plotting
        return X[:,::-1], y_true

    def kmeans_sklearn(self,x):
        kmeans = KMeans(self.K,random_state=0)
        return kmeans.fit(x).predict(x)


    def plot_kmeans(kmeans,labels, X, n_clusters=4, rseed=0, ax=None):
        # plot the input data
        ax = ax or plt.gca()
        ax.axis('equal')
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

        # plot the representation of the KMeans model
        centers = kmeans.cluster_centers_
        radii = [cdist(X[labels == i], [center]).max()
                             for i, center in enumerate(centers)]
        for c, r in zip(centers, radii):
            ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))


    def kde_sklearn(self,x, x_grid, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scikit-learn"""
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)

    def plot_kde(self, x_grid, pdf, pdf_true):
        """Function to plot results """
        fig,ax = plt.subplots(1,1,sharey=True,figsize=(13,5))
        fig.subplots_adjust(wspace=0)

        ax.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
        ax.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
        ax.set_title("GMM PLOT")

        if self.plot:
            plt.show()
        else:
            plt.savefig("GMM_Plot{}_Points".format(self.N))

    def gmm_sklearn(self,x):
        gmm = GMM(n_components=self.K).fit(x)
        labels = gmm.predict(x)
        probs = gmm.predict_proba(x)
        return labels, probs

    def draw_ellipse(self,position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))

    def plot_gmm(self,gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        ax.axis('equal')

        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)



if __name__=="__main__":
    cluster = Cluster(plot=False,K=4)
    x_grid,X,pdf_true = cluster._init_gauss()
    x,_ = cluster._create_blobs()
    #kde
    pdf = cluster.kde_sklearn(X, x_grid)
    cluster.plot_kde(x_grid,pdf,pdf_true)
    #kmeans
    kmeans_labels = cluster.kmeans_sklearn(x)
    #Gaussian Mixture
    gmm_labels,gmm_prob = cluster.gmm_sklearn(x)
    pdb.set_trace()




