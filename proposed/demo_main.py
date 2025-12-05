"""
demo_main.py

Minimal end-to-end check for:
  - GaussianCopula (on-manifold sampling)
  - OODGate + conformal threshold
  - IDLimeWrapper (ID-LIME)
  - IDKernelShap (ID-KernelSHAP)
  - Rank consistency & Infidelity comparison vs vanilla LIME / KernelSHAP
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
import shap
from copula import GaussianCopula
from ood_gate import OODGate, GateConfig
from id_explainers import IDLimeWrapper, IDKernelShap

def build_toy_dataset(n_samples: int = 500, n_features: int = 6, random_state: int = 42):
    """Create a small, linearly-separable-ish binary classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=random_state,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    class_names = ["class0", "class1"]
    return X, y, feature_names, class_names


def train_model(X_train, y_train):
    """Train a simple Logistic Regression classifier."""
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    return clf

def lime_to_vector(exp, n_features: int, label: int = 1) -> np.ndarray:
    """
    Convert a LimeTabular explanation into a dense vector of length d
    for the given class label.
    """
    vec = np.zeros(n_features, dtype=float)
    # exp.as_map()[label] → list of (feature_index, weight)
    m = exp.as_map()[label]
    for j, w in m:
        vec[j] = w
    return vec


def shap_to_vector(shap_values, d, positive_class=1):
    """
    Always return a vector of length d.

    Handles:
    - list of per-class arrays (each (1, d) or (d,))
    - array of shape (1, d, K)  <-- what we're seeing now
    - array of shape (1, d)
    - array of shape (d,)
    """
    # Case 1: list of per-class arrays
    if isinstance(shap_values, list):
        arr = np.array(shap_values[positive_class]).reshape(-1)
        return arr[:d]

    # Case 2: numpy array
    if isinstance(shap_values, np.ndarray):
        # (1, d, K) -> pick class dimension
        if shap_values.ndim == 3:
            # shape (1, d, K)
            _, d_shap, k = shap_values.shape
            # sanity: use min(d, d_shap)
            d_use = min(d, d_shap)
            vec = shap_values[0, :d_use, positive_class]
            return vec.reshape(-1)

        # (1, d) -> flatten
        if shap_values.ndim == 2:
            # typical: (1, d)
            if shap_values.shape[0] == 1:
                return shap_values.reshape(-1)[:d]
            # or (d, 1)
            if shap_values.shape[1] == 1:
                return shap_values.reshape(-1)[:d]
            # fallback: first row
            return shap_values[0, :d].reshape(-1)

        # (d,) -> already a vector
        if shap_values.ndim == 1:
            return shap_values[:d]

    raise ValueError(
        f"Unexpected SHAP format: {type(shap_values)}, shape={np.shape(shap_values)}"
    )

def rank_consistency(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """
    Rank consistency between two attribution vectors using Spearman rank corr
    on absolute attributions (so sign flips don't dominate).
    """
    a = np.abs(phi_a)
    b = np.abs(phi_b)
    return float(spearmanr(a, b).correlation)


def compute_infidelity(
    model,
    x: np.ndarray,
    phi: np.ndarray,
    n_mc: int = 50,
    noise_scale: float = 0.1,
    class_idx: int = 1,
) -> float:
    """
    Infidelity metric (Yeh et al.) approximated via Monte Carlo:

      INFD(phi, f, x) = E_I [ ( I^T phi(x) - ( f(x) - f(x - I) ) )^2 ]

    Here:
      - f(x) is the predicted probability for `class_idx`
      - I is drawn from N(0, noise_scale^2 I_d)
    """
    d = x.shape[0]

    def f(z: np.ndarray) -> float:
        p = model.predict_proba(z.reshape(1, -1))[0, class_idx]
        return float(p)

    f_x = f(x)
    errs = []
    for _ in range(n_mc):
        I = np.random.normal(loc=0.0, scale=noise_scale, size=d)
        lhs = float(I @ phi)
        f_x_minus = f(x - I)
        rhs = f_x - f_x_minus
        errs.append((lhs - rhs) ** 2)
    return float(np.mean(errs))

def main():
    X, y, feature_names, class_names = build_toy_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = train_model(X_train, y_train)
    print("[INFO] Trained Logistic Regression model.")
    print("[INFO] Classes:", model.classes_)

    # basic splits for copula / gate
    X_ref = X_train[:150]
    X_cal = X_train[150:350]
    X_eval = X_test
    x_test = X_eval[0]
    d = X.shape[1]

    gc = GaussianCopula().fit(X_ref)
    print("[INFO] Fitted GaussianCopula on reference data.")

    config = GateConfig(alpha=0.5)
    gate = OODGate(config)
    gate.fit(
        X_ref=X_ref,
        logits_ref=model.predict_proba(X_ref),
        X_cal=X_cal,
        logits_cal=model.predict_proba(X_cal),
        copula_model=gc,
    )
    print("[INFO] Fitted OODGate with conformal threshold tau =", gate.tau_)


    lime_base = LimeTabularExplainer(
        X_ref,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
    )

    id_lime = IDLimeWrapper(lime_base, model, gc, gate)

    vanilla_lime_exp = lime_base.explain_instance(
        x_test,
        model.predict_proba,
        num_features=d,
        num_samples=1000,
    )

    id_lime_exp = id_lime.explain_instance(
        x_test,
        label=1,
        num_samples=1000,
    )

    phi_lime_vanilla = lime_to_vector(vanilla_lime_exp, d, label=1)
    phi_lime_id = lime_to_vector(id_lime_exp, d, label=1)

    print("\n================= LIME explanations (class 1) =================")
    print("Vanilla LIME feature importances:")
    for name, w in zip(feature_names, phi_lime_vanilla):
        print(f"{name:25s} {w:+.4f}")

    print("\nID-LIME feature importances:")
    for name, w in zip(feature_names, phi_lime_id):
        print(f"{name:25s} {w:+.4f}")

    lime_rank_cons = rank_consistency(phi_lime_vanilla, phi_lime_id)
    print(f"\n[METRIC] LIME vs ID-LIME rank consistency: {lime_rank_cons:.4f}")

    inf_lime_vanilla = compute_infidelity(model, x_test, phi_lime_vanilla, n_mc=50)
    inf_lime_id = compute_infidelity(model, x_test, phi_lime_id, n_mc=50)
    print(f"[METRIC] LIME vanilla infidelity: {inf_lime_vanilla:.6f}")
    print(f"[METRIC] ID-LIME infidelity     : {inf_lime_id:.6f}")


    explainer_vanilla_kshap = shap.KernelExplainer(
        lambda X_: model.predict_proba(X_),
        X_ref,
        link="identity",
    )

    shap_vals_vanilla = explainer_vanilla_kshap.shap_values(
        x_test.reshape(1, -1),
        nsamples=200,
    )
    phi_kshap_vanilla = shap_to_vector(shap_vals_vanilla, d, positive_class=1)

    id_kshap = IDKernelShap(model, gc, gate, link="identity")
    shap_vals_id = id_kshap.explain_instance(
        x_test,
        nsamples=200,
    )
    phi_kshap_id = shap_to_vector(shap_vals_id, d, positive_class=1)

    print("\n================= KernelSHAP (positive class) =================")
    print("Vanilla KernelSHAP:")
    for name, val in zip(feature_names, phi_kshap_vanilla):
        print(f"{name:25s} {val:+.4f}")

    print("\nID-KernelSHAP:")
    for name, val in zip(feature_names, phi_kshap_id):
        print(f"{name:25s} {val:+.4f}")

    kshap_rank_cons = rank_consistency(phi_kshap_vanilla, phi_kshap_id)
    print(f"\n[METRIC] KernelSHAP vs ID-KernelSHAP rank consistency: {kshap_rank_cons:.4f}")

    inf_kshap_vanilla = compute_infidelity(model, x_test, phi_kshap_vanilla, n_mc=50)
    inf_kshap_id = compute_infidelity(model, x_test, phi_kshap_id, n_mc=50)
    print(f"[METRIC] KernelSHAP vanilla infidelity: {inf_kshap_vanilla:.6f}")
    print(f"[METRIC] ID-KernelSHAP infidelity     : {inf_kshap_id:.6f}")

    print("\n[INFO] Demo + metrics complete.")

if __name__ == "__main__":
    main()
