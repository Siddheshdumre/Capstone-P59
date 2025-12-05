import shap
import numpy as np
from typing import Any
from sklearn.metrics.pairwise import pairwise_distances


class IDLimeWrapper:
    """
    ID-LIME: On-manifold + OOD-aware version of LIME that does NOT rely
    on private sampling APIs. It talks directly to LimeBase.explain_instance_with_data.
    """

    def __init__(self, base_lime: Any, model: Any, copula_model: Any, ood_gate: Any):
        """
        Parameters
        ----------
        base_lime : LimeTabularExplainer (your local version under explainers.lime)
        model : object with predict_proba(X) or predict(X)
        copula_model : fitted GaussianCopula instance
        ood_gate : fitted OODGate instance
        """
        self.base_lime = base_lime
        self.model = model
        self.copula = copula_model
        self.ood_gate = ood_gate

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Safe wrapper around model.predict_proba / predict."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            pred = self.model.predict(X)
            return np.asarray(pred, dtype=float)

    def explain_instance(
        self,
        x,
        label: int,
        num_samples: int = 5000,
        oversample_factor: float = 2.0,
        num_features: int | None = None,
        **kwargs,
    ):
        """
        ID-LIME explanation for a single instance.

        Pipeline:
          1. Sample local neighborhood via Gaussian Copula.
          2. Score & filter with OOD gate (conformal acceptance).
          3. Call LimeBase.explain_instance_with_data using only
             conformally accepted samples.

        Parameters
        ----------
        x : array-like, shape (d,)
            Instance to explain.
        label : int
            Class index to explain.
        num_samples : int
            Target number of perturbations passed to LIME.
        oversample_factor : float
            Factor to oversample before gating to compensate for rejections.
        num_features : int or None
            Number of features to show in the explanation. If None, use all.

        Returns
        -------
        explanation :
            LIME Explanation object for the given label.
        """
        x = np.asarray(x, dtype=float).ravel()

        # 1. copula-based perturbations
        n_raw = int(num_samples * oversample_factor)
        X_pert = self.copula.sample_local(x, n_samples=n_raw)

        # 2. model outputs for gate
        probs = self._predict_proba(X_pert)
        logits = np.log(probs + 1e-12)

        mask = self.ood_gate.accept(X_pert, logits)

        # 3. select samples after gating
        if mask.sum() < num_samples // 4:
            # gate too strict → take best by score
            scores = self.ood_gate.score(X_pert, logits)
            idx = np.argsort(scores)[:num_samples]
            X_used = X_pert[idx]
            probs_used = probs[idx]
        else:
            X_used = X_pert[mask][:num_samples]
            probs_used = probs[mask][:num_samples]

        # 4. compute distances between samples and x
        metric = getattr(self.base_lime, "distance_metric", "euclidean")
        distances = pairwise_distances(
            X_used,
            x.reshape(1, -1),
            metric=metric,
        ).ravel()

        # 5. call LimeBase directly
        feature_selection = getattr(self.base_lime, "feature_selection", "auto")
        k = num_features or X_used.shape[1]

        explanation = self.base_lime.base.explain_instance_with_data(
            X_used,
            probs_used,
            distances,
            label,
            k,
            feature_selection=feature_selection,
        )

        result = self.base_lime.base.explain_instance_with_data(
            X_used,
            probs_used,
            distances,
            label,
            k,
            feature_selection=feature_selection,
        )

        if hasattr(result, "as_map"):
            return result

        if isinstance(result, tuple) and hasattr(result[0], "as_map"):
            return result[0]

        return self.base_lime.explain_instance(
            x,
            self._predict_proba,
            num_features=k
        )


class IDKernelShap:
    """
    ID-KernelSHAP: builds a local KernelExplainer on top of
    on-manifold, OOD-filtered background samples.

    For each instance:
      1. Generate local neighborhood with Gaussian Copula.
      2. Filter via OODGate + conformal threshold.
      3. Use accepted samples as background for shap.KernelExplainer.
    """

    def __init__(
        self,
        model: Any,
        copula_model: Any,
        ood_gate: Any,
        link: str = "identity",
    ):
        """
        Parameters
        ----------
        model : object with predict_proba(X) or predict(X)
        copula_model : fitted GaussianCopula instance
        ood_gate : fitted OODGate instance
        link : str, default "identity"
            Link function for shap.KernelExplainer (e.g., "logit" for probabilities).
        """
        self.model = model
        self.copula = copula_model
        self.ood_gate = ood_gate
        self.link = link

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Safe wrapper: returns probabilities or scores."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            pred = self.model.predict(X)
            return np.asarray(pred, dtype=float)

    def _model_f(self, X: np.ndarray) -> np.ndarray:
        """
        Model function passed to KernelExplainer.

        Returns class probabilities / scores; KernelSHAP will
        compute per-class SHAP values.
        """
        X = np.asarray(X, dtype=float)
        return self._predict_proba(X)

    def explain_instance(
        self,
        x,
        nsamples: int = 1000,
        n_background: int = 500,
        oversample_factor: float = 2.0,
        **kernel_kwargs,
    ):
        """
        Compute ID-KernelSHAP values for a single instance.

        Pipeline:
          1. Sample local neighborhood via Gaussian Copula.
          2. Score & filter with OOD gate.
          3. Use accepted points as local background for KernelExplainer.
          4. Run SHAP on x with nsamples draws.

        Parameters
        ----------
        x : array-like, shape (d,)
            Instance to explain.
        nsamples : int
            Number of SHAP kernel samples.
        n_background : int
            Number of background points (after gating) to give to SHAP.
        oversample_factor : float
            Factor by which to oversample before gating.
        **kernel_kwargs :
            Extra kwargs forwarded to shap.KernelExplainer.shap_values.

        Returns
        -------
        shap_values :
            SHAP values for x. For a K-class model, this is typically a
            list of length K, each entry shape (1, d).
        """
        x = np.asarray(x, dtype=float).ravel()

        # 1. copula-based perturbations around x
        n_raw = int(n_background * oversample_factor)
        X_pert = self.copula.sample_local(x, n_samples=n_raw)

        # 2. OOD gate
        probs = self._predict_proba(X_pert)
        logits = np.log(probs + 1e-12)

        mask = self.ood_gate.accept(X_pert, logits)

        if mask.sum() < n_background // 4:
            # too strict -> keep top n_background by score
            scores = self.ood_gate.score(X_pert, logits)
            idx = np.argsort(scores)[:n_background]
            background = X_pert[idx]
        else:
            background = X_pert[mask][:n_background]

        # 3. Local KernelExplainer on accepted on-manifold neighborhood
        explainer = shap.KernelExplainer(
            self._model_f,
            background,
            link=self.link,
        )

        shap_values = explainer.shap_values(
            x.reshape(1, -1),
            nsamples=nsamples,
            **kernel_kwargs,
        )
        return shap_values
