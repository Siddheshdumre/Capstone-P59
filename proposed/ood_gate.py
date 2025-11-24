# ood_gate.py
import numpy as np
from dataclasses import dataclass
from typing import Dict


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Energy score from logits as in Energy-based OOD detection.
    Lower energy → more in-distribution.
    """
    logits = np.asarray(logits, dtype=float)
    # shape (n, K) or (K,)
    if logits.ndim == 1:
        logits = logits[None, :]
    scaled = logits / temperature
    # logsumexp per row
    max_ = np.max(scaled, axis=1, keepdims=True)
    lse = max_ + np.log(np.sum(np.exp(scaled - max_), axis=1, keepdims=True))
    energy = -temperature * lse.squeeze(-1)
    return energy  # shape (n,)


def mahalanobis_distance(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """
    Classic Mahalanobis distance in feature space.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    diff = X - mean
    md2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return np.sqrt(md2)

@dataclass
class GateConfig:
    alpha: float = 0.15
    w_copula: float = 0.1
    w_energy: float = 0.05
    w_maha: float = 0.05

class OODGate:
    """
    OOD gate that:
      - computes 3 scores (copula, energy, Mahalanobis)
      - normalizes them using calibration stats
      - combines to a scalar non-conformity score
      - learns a conformal threshold for given alpha
    """
    def __init__(self, config: GateConfig):
        self.config = config
        self.copula_ = None

        # feature-space stats for Mahalanobis
        self.mean_ = None
        self.inv_cov_ = None

        # calibration stats
        self.score_mean_ = None
        self.score_std_ = None
        self.tau_ = None

    def fit_reference(
        self,
        X_ref: np.ndarray,
        logits_ref: np.ndarray,
        copula_model,
    ):
        """
        Fit reference stats for:
          - Mahalanobis (mean, covariance)
          - copula (already fitted externally, passed as argument)
        """
        X_ref = np.asarray(X_ref, dtype=float)
        self.mean_ = X_ref.mean(axis=0)
        cov = np.cov(X_ref, rowvar=False)
        cov += 1e-3 * np.eye(cov.shape[0])
        self.inv_cov_ = np.linalg.inv(cov)
        self.copula_ = copula_model

    def _compute_scores(
        self,
        X: np.ndarray,
        logits: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute raw scores (before normalization/combination).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        logits = np.asarray(logits, dtype=float)
        if logits.ndim == 1:
            logits = logits[None, :]

        # Copula negative log-likelihood (higher → more OOD)
        cop_ll = np.array([self.copula_.log_likelihood(x) for x in X])
        cop_score = -cop_ll

        # Energy score (higher energy → more OOD)
        en_score = energy_score(logits)

        # Mahalanobis distance in feature space
        md_score = mahalanobis_distance(X, self.mean_, self.inv_cov_)

        return {
            "copula": cop_score,
            "energy": en_score,
            "maha": md_score,
        }

    def _combine_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine normalized scores into a single scalar.
        """
        s_c = scores["copula"]
        s_e = scores["energy"]
        s_m = scores["maha"]

        stacked = np.vstack([s_c, s_e, s_m]).T  # (n, 3)
        # z-score normalize using calibration stats
        normed = (stacked - self.score_mean_) / (self.score_std_ + 1e-12)

        w = np.array(
            [self.config.w_copula, self.config.w_energy, self.config.w_maha],
            dtype=float,
        )
        combined = normed @ w
        return combined  # shape (n,)

    def fit_conformal(self, X_cal: np.ndarray, logits_cal: np.ndarray):
        """
        Fit conformal threshold tau using calibration set.
        """
        raw = self._compute_scores(X_cal, logits_cal)
        stacked = np.vstack(
            [raw["copula"], raw["energy"], raw["maha"]]
        ).T  # (n, 3)

        self.score_mean_ = stacked.mean(axis=0)
        self.score_std_ = stacked.std(axis=0)

        cal_scores = self._combine_scores(raw)  # (n,)

        # conformal quantile: non-conformity → larger = worse
        alpha = self.config.alpha
        n = cal_scores.shape[0]
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        k = np.clip(k, 0, n - 1)
        tau = np.partition(np.sort(cal_scores), k)[k]
        self.tau_ = float(tau)

    def fit(
        self,
        X_ref: np.ndarray,
        logits_ref: np.ndarray,
        X_cal: np.ndarray,
        logits_cal: np.ndarray,
        copula_model,
    ):
        """
        Convenience wrapper:
          1. fit_reference on (X_ref, logits_ref)
          2. fit_conformal on calibration subset
        """
        self.fit_reference(X_ref, logits_ref, copula_model)
        self.fit_conformal(X_cal, logits_cal)

    def score(self, X: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """Return combined non-conformity score."""
        raw = self._compute_scores(X, logits)
        return self._combine_scores(raw)

    def accept(self, X: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """
        Boolean mask of accepted samples (True = in-distribution enough).
        """
        s = self.score(X, logits)
        return s <= self.tau_
