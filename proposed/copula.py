# gaussian_copula.py
import numpy as np
from scipy import stats
from numpy.linalg import slogdet, inv


class GaussianCopula:
    """
    Gaussian Copula model for:
      - on-manifold sampling
      - latent-space log-likelihood (for OOD scoring)
    """

    def __init__(self, eps=1e-6, cov_shrink=1e-3):
        self.eps = eps
        self.cov_shrink = cov_shrink
        self.X_train_ = None          # (n, d)
        self.sorted_cols_ = None      # list of sorted values per feature
        self.corr_ = None             # (d, d) correlation matrix in z-space
        self.corr_inv_ = None
        self.logdet_corr_ = None
        self.n_features_ = None
        self.n_train_ = None

    # ---------- fitting ----------

    def fit(self, X: np.ndarray):
        """
        Fit Gaussian copula on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.X_train_ = X
        self.n_train_, self.n_features_ = n, d

        # store sorted columns for inverse CDF
        self.sorted_cols_ = [np.sort(X[:, j]) for j in range(d)]

        # PIT transform: rank -> uniform (0,1)
        ranks = np.empty_like(X, dtype=float)
        for j in range(d):
            ranks[:, j] = stats.rankdata(X[:, j], method="average")

        u = (ranks - 0.5) / (n + 1.0)
        u = np.clip(u, self.eps, 1 - self.eps)

        # Gaussianize
        z = stats.norm.ppf(u)

        # correlation in z-space (rows = samples)
        corr = np.cov(z, rowvar=False)

        # shrinkage for numerical stability
        corr = (1 - self.cov_shrink) * corr + self.cov_shrink * np.eye(d)

        self.corr_ = corr
        sign, logdet = slogdet(corr)
        if sign <= 0:
            raise RuntimeError("Copula correlation matrix not PD after shrinkage.")
        self.logdet_corr_ = logdet
        self.corr_inv_ = inv(corr)
        return self

    def _x_to_u(self, x: np.ndarray) -> np.ndarray:
        """Map a single x to PIT u (0,1)^d using empirical CDFs."""
        x = np.asarray(x, dtype=float)
        u = np.empty_like(x, dtype=float)
        for j in range(self.n_features_):
            col = self.sorted_cols_[j]
            # how many training points <= x_j
            k = np.searchsorted(col, x[j], side="right")
            u[j] = (k + 0.5) / (self.n_train_ + 1.0)
        u = np.clip(u, self.eps, 1 - self.eps)
        return u

    def _u_to_x(self, u: np.ndarray) -> np.ndarray:
        """Inverse PIT: from u in (0,1)^d back to approximate x."""
        u = np.asarray(u, dtype=float)
        x = np.empty_like(u, dtype=float)
        for j in range(self.n_features_):
            col = self.sorted_cols_[j]
            idx = int(np.floor(u[j] * (self.n_train_ - 1)))
            idx = np.clip(idx, 0, self.n_train_ - 1)
            x[j] = col[idx]
        return x

    def _x_to_z(self, x: np.ndarray) -> np.ndarray:
        u = self._x_to_u(x)
        z = stats.norm.ppf(u)
        return z

    def log_likelihood(self, x: np.ndarray) -> float:
        """
        Latent Gaussian log-likelihood under the copula (up to additive const).
        """
        z = self._x_to_z(x)
        # MVN(0, corr) logpdf
        quad = z @ self.corr_inv_ @ z
        d = self.n_features_
        logpdf = -0.5 * (quad + self.logdet_corr_ + d * np.log(2 * np.pi))
        return float(logpdf)

    def sample_local(self, x: np.ndarray, n_samples: int = 5000,
                     scale: float = 0.2) -> np.ndarray:
        """
        Sample local on-manifold perturbations around x.

        We center a Gaussian in z-space at z(x) and reuse corr_ as shape.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
        n_samples : int
        scale : float
            Scaling factor for covariance (smaller → tighter neighborhood).

        Returns
        -------
        X_s : np.ndarray of shape (n_samples, n_features)
        """
        z0 = self._x_to_z(x)
        cov_local = scale * self.corr_
        z_samples = np.random.multivariate_normal(mean=z0, cov=cov_local,
                                                  size=n_samples)
        # back to x via Gaussian -> uniform -> empirical inverse CDF
        X_s = np.empty_like(z_samples)
        for i in range(n_samples):
            u = stats.norm.cdf(z_samples[i, :])
            X_s[i, :] = self._u_to_x(u)
        return X_s
