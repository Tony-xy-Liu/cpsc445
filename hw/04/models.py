from typing import Any
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

def _kmeans_init(N, K, X, seed):
    kmeans = KMeans(n_clusters=K, n_init="auto", random_state=seed).fit(X)
    lambdas = np.zeros(shape=K)
    pi = np.zeros(shape=K)
    for k in range(K):
        lbls: Any = kmeans.labels_
        selector = [i for i, z in enumerate(lbls) if z == k]
        lambdas[k] = X[selector].mean()
        pi[k] = len(selector)/N

    return pi, lambdas

class PoissonEM:
    def __init__(
            self,
            alphas: list[float], nu: float, beta: float, 
            iterations: int, weighted_lambdas=True,
            initial_lambdas = None,
            random_seed:int=0
        ) -> None:
        
        self._seed = random_seed
        self.num_clusters = len(alphas)
        self.alphas = alphas
        self.nu = nu
        self.beta = beta
        self.iterations = iterations
        self._adjust_pi = weighted_lambdas
        if initial_lambdas is not None:
            self.lambdas = initial_lambdas

    def Fit(self, x: np.ndarray, silent=False):
        try: from sklearnex import patch_sklearn; patch_sklearn()
        except ModuleNotFoundError:
            pass

        X = x
        K, N = self.num_clusters, len(x)
        SUMMED_ALPHA = np.sum(self.alphas)
        if not self._adjust_pi: # this check is incomplete but works for the purpose of the hw
            self.pi = np.ones(shape=K)/K
        else:
            self.pi, self.lambdas = _kmeans_init(N, K, X, self._seed)

        # em
        z_prob = np.zeros(shape=(N, K))
        ll_last_component = np.zeros(shape=z_prob.shape)
        _gammas = np.zeros(shape=K)
        self.log_lls = np.zeros(shape=self.iterations)
        self.lambda_history = np.zeros(shape=(self.iterations+1, K))
        self.lambda_history[0] = [l for l in self.lambdas]
        for it in range(self.iterations):

            # e-step
            for k in range(K):
                _gammas[k] = stats.gamma.pdf(self.lambdas[k], self.nu)
                fthetas = stats.poisson.pmf(X[:, 0], self.lambdas[k])
                ll_last_component[:, k] = z_prob[:, k] = self.pi[k]*fthetas
            z_prob = (z_prob.T/z_prob.sum(axis=1)).T

            # log likelihood & progress
            log_ll = np.log(stats.dirichlet.pdf(self.pi, self.alphas))\
                + np.log(_gammas).sum()\
                + np.log(ll_last_component.sum(axis=1)).sum()
            self.log_lls[it] = log_ll
            _delta = log_ll - self.log_lls[it-1] if it >0 else 0
            _progress = f"{it+1}: log lik.: {log_ll:0.3f} | delta: {max(_delta, 0):0.6f}"
            if not silent: print(f"{_progress}{' '*(50-len(_progress))}", end='\r' if it<self.iterations-1 else '\n')

            # m-step
            for k in range(K):
                summed_prob_k = z_prob[:, k].sum()
                if self._adjust_pi: self.pi[k] = (self.alphas[k] - 1 + summed_prob_k) / (SUMMED_ALPHA - K + N)
                prior1, prior2 = (1, 2) if self._adjust_pi else (0, 0)
                self.lambdas[k] = ((z_prob[:, k]*X[:, 0]).sum()+prior1) / (summed_prob_k+prior2)
                self.lambda_history[it+1, k] = self.lambdas[k]

class PoissonGibbs:
    def __init__(
            self,
            alphas: list[float], nu: float, beta: float,
            iterations: int,
            random_seed:int=0
        ) -> None:
        
        self._seed = random_seed
        self.num_clusters = len(alphas)
        self.alphas = alphas
        self.nu = nu
        self.beta = beta
        self.iterations = iterations

    def Fit(self, x: np.ndarray, silent=False):
        try: from sklearnex import patch_sklearn; patch_sklearn()
        except ModuleNotFoundError:
            pass

        X = x
        K, N = self.num_clusters, len(X)
        NU, BETA = self.nu, self.beta
        _pi, _lambdas = _kmeans_init(N, K, X, self._seed)
        self.lambdas = np.zeros(shape=(self.iterations+1, K)); self.lambdas[0] = _lambdas
        self.pi = np.zeros(shape=(self.iterations+1, K)); self.pi[0] = _pi

        _z_weights = np.zeros(shape=K)
        _z_prob = np.zeros(shape=K)
        _z_sums = np.zeros(shape=K)
        _z_counts = np.zeros(shape=K)
        for it in range(self.iterations):
            if not silent: print(f"iteration: {it+1}", end='\r')

            # sample Z
            _z_sums *= 0
            _z_counts *= 0
            for x in X[:, 0]: # x_i in N
                _z_prob = self.pi[it] * stats.poisson.pmf([x]*K, self.lambdas[it])
                _z_prob /= _z_prob.sum()
                theta = stats.multinomial(1, _z_prob) # 1 trial, prob. success is Z
                z = [i for i, z in enumerate(theta.rvs(size=1)[0]) if z > 0][0]

                _z_sums[z] += x
                _z_counts[z] += 1

            # sample pi
            for k in range(K):
                self.lambdas[it+1, k] = stats.gamma(NU + _z_sums[k]).rvs(size=1)[0]/(BETA + _z_counts[k])
                _z_weights[k] = self.alphas[k] + _z_counts[k]
            self.pi[it+1] = stats.dirichlet.rvs(_z_weights, size=1)[0]

