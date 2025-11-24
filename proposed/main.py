"""
demo_id_explainers.py

Minimal end-to-end check for:
  - GaussianCopula (on-manifold sampling)
  - OODGate + conformal threshold
  - IDLimeWrapper (ID-LIME)
  - IDKernelShap (ID-KernelSHAP)

"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer
from gaussian_copula import GaussianCopula
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


def main():
    X, y, feature_names, class_names = build_toy_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = train_model(X_train, y_train)
    print("[INFO] Trained Logistic Regression model.")
    print("[INFO] Classes:", model.classes_)

    X_ref = X_train[:150]
    X_cal = X_train[150:350]
    X_eval = X_test
    x_test = X_eval[0]

    gc = GaussianCopula().fit(X_ref)
    print("[INFO] Fitted GaussianCopula on reference data.")

    config = GateConfig(alpha=0.05)
    gate = OODGate(config)
    gate.fit(
        X_ref=X_ref,
        logits_ref=model.predict_proba(X_ref),
        X_cal=X_cal,
        logits_cal=model.predict_proba(X_cal),
        copula_model=gc,
    )
    print("[INFO] Fitted OODGate with conformal threshold tau =", gate.tau_)

    logits_eval = model.predict_proba(X_eval)
    accept_eval = gate.accept(X_eval, np.log(logits_eval + 1e-12))

    rng = np.random.RandomState(123)
    X_ood = rng.normal(loc=0.0, scale=10.0, size=X_eval.shape)
    logits_ood = model.predict_proba(X_ood)
    accept_ood = gate.accept(X_ood, np.log(logits_ood + 1e-12))

    lime_base = LimeTabularExplainer(
        X_ref,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
    )

    id_lime = IDLimeWrapper(lime_base, model, gc, gate)

    lime_exp = id_lime.explain_instance(
        x_test,
        label=1,
        num_samples=1000,
    )

    print("\n================ ID-LIME explanation (class1) ================")
    print("ID-LIME raw output:", lime_exp)

    id_kshap = IDKernelShap(model, gc, gate, link="identity")

    shap_values = id_kshap.explain_instance(
        x_test,
        nsamples=200,
    )

    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv_pos = np.array(shap_values[1]).reshape(-1)
    elif isinstance(shap_values, list) and len(shap_values) == 1:
        sv_pos = np.array(shap_values[0]).reshape(-1)
    elif isinstance(shap_values, np.ndarray):
        sv_pos = shap_values.reshape(-1)

    else:
        raise ValueError(f"Unexpected SHAP output format: {type(shap_values)}")

    print("\n===== ID-KernelSHAP (positive class) =====")
    for name, val in zip(feature_names, sv_pos):
        print(f"{name:25s} {val:+.4f}")

    print("\n[INFO] Demo complete.")

if __name__ == "__main__":
    main()
