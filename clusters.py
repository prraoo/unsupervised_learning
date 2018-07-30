import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from sklearn.neighbors import KernelDensity

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

    def kde_sklearn(self,x, x_grid, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scikit-learn"""
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)

    def _plot(self, x_grid, pdf, pdf_true):
        """Function to plot results """
        fig,ax = plt.subplots(1,1,sharey=True,figsize=(13,5))
        fig.subplots_adjust(wspace=0)

        ax.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
        ax.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
        ax.set_title("GMM PLOT")
        plt.show()


if __name__=="__main__":
    cluster = Cluster()
    x_grid,X,pdf_true = cluster._init_gauss()
    pdf = cluster.kde_sklearn(X, x_grid)
    cluster._plot(x_grid,pdf,pdf_true)



