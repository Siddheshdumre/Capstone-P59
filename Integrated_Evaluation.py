import numpy as np
import pandas as pd
import sklearn

from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

from copula import GaussianCopula
from ood_gate import OODGate, GateConfig
from id_explainers import IDLimeWrapper, IDKernelShap


# -----------------------------
# 1. DATA LOADING + PREPROCESS
# -----------------------------

TARGET = "income"        # target column after renaming
POS_LABEL = ">50K"       # positive label
ADULT_CSV_PATH = "/content/Capstone-P59/data/adult/adult/adult.data"


def build_adult_dataset(
    csv_path: str = ADULT_CSV_PATH,
    target_col: str = TARGET,
    pos_label: str = POS_LABEL,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load Adult dataset (raw .data file), assign proper column names,
    build X/y, and return raw + encoded splits + preprocessor + feature names.
    """
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",          # original 'sex', renamed to gender
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",          # target
    ]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=col_names,
        skipinitialspace=True,  # strips leading spaces
    )

    # Binary label
    y = (df[target_col] == pos_label).astype(int)
    X = df.drop(columns=[target_col])

    # numeric vs categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # sklearn version handling
    if sklearn.__version__ >= "1.4":
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", cat_encoder),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train_enc = preproc.fit_transform(X_train_raw)
    X_test_enc = preproc.transform(X_test_raw)

    # Encoded feature names
    try:
        feature_names_enc = preproc.get_feature_names_out()
    except AttributeError:
        num_names = num_cols
        cat_names = (
            preproc.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(cat_cols)
            .tolist()
        )
        feature_names_enc = num_names + cat_names

    return (
        X_train_raw,
        X_test_raw,
        X_train_enc,
        X_test_enc,
        y_train.values,
        y_test.values,
        preproc,
        feature_names_enc,
    )


# -----------------------------
# 2. MODEL TRAINING
# -----------------------------

def train_logreg_on_encoded(X_train_enc: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    # Slightly reduced max_iter for speed, still safe on Adult
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train_enc, y_train)
    return clf


# -----------------------------
# 3. ATTRIBUTION UTILS
# -----------------------------

def lime_to_vector(exp, n_features: int, label: int = 1) -> np.ndarray:
    """
    Convert a LIME explanation into a dense vector of length d.
    """
    vec = np.zeros(n_features, dtype=float)
    m = exp.as_map()[label]
    for j, w in m:
        vec[j] = w
    return vec


def shap_to_vector(shap_values, d: int, positive_class: int = 1) -> np.ndarray:
    """
    Always return a vector of length d from SHAP outputs.
    Handles:
      - list of per-class arrays
      - (1, d, K)
      - (1, d)
      - (d,)
    """
    if isinstance(shap_values, list):
        arr = np.array(shap_values[positive_class]).reshape(-1)
        return arr[:d]

    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            _, d_shap, k = shap_values.shape
            d_use = min(d, d_shap)
            vec = shap_values[0, :d_use, positive_class]
            return vec.reshape(-1)

        if shap_values.ndim == 2:
            if shap_values.shape[0] == 1:
                return shap_values.reshape(-1)[:d]
            if shap_values.shape[1] == 1:
                return shap_values.reshape(-1)[:d]
            return shap_values[0, :d].reshape(-1)

        if shap_values.ndim == 1:
            return shap_values[:d]

    raise ValueError(
        f"Unexpected SHAP format: {type(shap_values)}, shape={np.shape(shap_values)}"
    )


def rank_consistency(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """
    Spearman rank correlation between |phi_a| and |phi_b|.
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
    Infidelity metric (Yeh et al.), Monte Carlo approximation.
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


# -----------------------------
# 4. ROAR / KAR HELPERS
# -----------------------------

def compute_global_ranking(attributions: np.ndarray) -> np.ndarray:
    """
    Given attributions matrix (n_samples, d),
    return feature indices sorted by mean |phi| descending.
    """
    importance = np.mean(np.abs(attributions), axis=0)
    return np.argsort(-importance)


def roar_kar_curves(
    X_train_enc: np.ndarray,
    y_train: np.ndarray,
    X_test_enc: np.ndarray,
    y_test: np.ndarray,
    feature_ranking: np.ndarray,
    fractions: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """
    Compute ROAR (remove top-k) and KAR (keep top-k) accuracy curves.
    Features are zeroed out as baseline after scaling (≈ mean).
    """
    d = X_train_enc.shape[1]
    roar_accs: List[float] = []
    kar_accs: List[float] = []

    baseline_value = 0.0  # standardized features → 0 ≈ mean

    for frac in fractions:
        k = max(1, int(frac * d))
        topk_idx = feature_ranking[:k]

        # ROAR: remove top-k
        X_train_roar = X_train_enc.copy()
        X_test_roar = X_test_enc.copy()
        X_train_roar[:, topk_idx] = baseline_value
        X_test_roar[:, topk_idx] = baseline_value

        clf_roar = train_logreg_on_encoded(X_train_roar, y_train)
        roar_accs.append(clf_roar.score(X_test_roar, y_test))

        # KAR: keep only top-k
        X_train_kar = np.zeros_like(X_train_enc) + baseline_value
        X_test_kar = np.zeros_like(X_test_enc) + baseline_value
        X_train_kar[:, topk_idx] = X_train_enc[:, topk_idx]
        X_test_kar[:, topk_idx] = X_test_enc[:, topk_idx]

        clf_kar = train_logreg_on_encoded(X_train_kar, y_train)
        kar_accs.append(clf_kar.score(X_test_kar, y_test))

    return roar_accs, kar_accs


# -----------------------------
# 5. OFF-MANIFOLD 3D VIZ UTILS
# -----------------------------

def generate_off_manifold_samples(
    X_train_enc: np.ndarray,
    n_samples: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    """
    Off-manifold sampling by breaking joint structure:
    each feature is sampled independently from its empirical marginal.
    """
    rng = np.random.default_rng(random_state)
    n, d = X_train_enc.shape
    off = np.zeros((n_samples, d), dtype=float)
    for j in range(d):
        idx = rng.integers(0, n, size=n_samples)
        off[:, j] = X_train_enc[idx, j]
    return off


def visualize_manifold_vs_off_3d(
    X_train_enc: np.ndarray,
    X_off: np.ndarray,
    n_points: int = 2000,
    title: str = "Adult – On-Manifold vs Off-Manifold (PCA-3D)",
    save_path: str | None = None,
):
    """
    3D PCA projection to compare on-manifold vs off-manifold samples.
    """
    n_train = min(n_points // 2, X_train_enc.shape[0])
    n_off = min(n_points // 2, X_off.shape[0])

    X_on_sub = X_train_enc[:n_train]
    X_off_sub = X_off[:n_off]

    X_combined = np.vstack([X_on_sub, X_off_sub])
    labels = np.array([0] * n_train + [1] * n_off)

    pca = PCA(n_components=3, random_state=0)
    X_3d = pca.fit_transform(X_combined)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X_3d[labels == 0, 0],
        X_3d[labels == 0, 1],
        X_3d[labels == 0, 2],
        s=8,
        alpha=0.4,
        label="On-manifold (train)",
    )
    ax.scatter(
        X_3d[labels == 1, 0],
        X_3d[labels == 1, 1],
        X_3d[labels == 1, 2],
        s=8,
        alpha=0.4,
        label="Off-manifold (prod-of-marginals)",
    )

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# -----------------------------
# 6. MAIN EXPERIMENT PIPELINE
# -----------------------------

def main():
    print("/content/Capstone-P59")

    (
        X_train_raw,
        X_test_raw,
        X_train_enc,
        X_test_enc,
        y_train,
        y_test,
        preproc,
        feature_names_enc,
    ) = build_adult_dataset()

    d = X_train_enc.shape[1]
    print(f"[INFO] X_train_enc: {X_train_enc.shape}, X_test_enc: {X_test_enc.shape}")

    # Base model
    base_model = train_logreg_on_encoded(X_train_enc, y_train)
    base_acc = base_model.score(X_test_enc, y_test)
    print(f"[INFO] Base Logistic Regression accuracy: {base_acc:.4f}")
    print("[INFO] Classes:", base_model.classes_)

    # Copula / OODGate / ID wrappers operate on encoded space
    X_ref = X_train_enc[:150]
    X_cal = X_train_enc[150:350]

    # Gaussian Copula
    copula = GaussianCopula().fit(X_ref)

    # Logits (or probs) for gate
    probs_ref = base_model.predict_proba(X_ref)
    probs_cal = base_model.predict_proba(X_cal)
    logits_ref = np.log(probs_ref + 1e-12)
    logits_cal = np.log(probs_cal + 1e-12)

    gate_cfg = GateConfig(alpha=0.1)
    gate = OODGate(config=gate_cfg)
    gate.fit(
        X_ref,
        logits_ref,
        X_cal,
        logits_cal,
        copula,  # copula_model
    )

    # Vanilla LIME explainer (encoded space)
    explainer_lime = LimeTabularExplainer(
        training_data=X_train_enc,
        feature_names=feature_names_enc,
        class_names=["<=50K", ">50K"],
        discretize_continuous=False,
        mode="classification",
    )

    # ID-LIME
    id_lime = IDLimeWrapper(
        explainer_lime,   # base_lime
        base_model,       # model
        copula,           # copula_model
        gate,             # ood_gate
    )

    # SHAP LinearExplainer for Logistic Regression (vanilla SHAP)
    background_size = 100
    background = shap.sample(X_train_enc, background_size, random_state=0)
    shap_explainer = shap.LinearExplainer(base_model, background)

    # ID-KernelSHAP
    id_shap = IDKernelShap(
        base_model,  # model
        copula,      # copula_model
        gate,        # ood_gate
        link="identity",
    )

    # -----------------------------------------
    # Attribution collection for N_eval samples
    # -----------------------------------------
    N_eval = min(100, X_test_enc.shape[0])  # reduced for speed
    X_eval = X_test_enc[:N_eval]

    attributions = {
        "LIME": [],
        "ID-LIME": [],
        "SHAP": [],
        "ID-KernelSHAP": [],
    }

    rank_lime_pairs: List[float] = []
    rank_shap_pairs: List[float] = []

    print(f"[INFO] Computing attributions on {N_eval} eval samples...")

    for i in range(N_eval):
        x = X_eval[i]

        # LIME (vanilla)
        exp_lime = explainer_lime.explain_instance(
            data_row=x,
            predict_fn=base_model.predict_proba,
            num_features=d,
        )
        phi_lime = lime_to_vector(exp_lime, n_features=d, label=1)
        attributions["LIME"].append(phi_lime)

        # ID-LIME
        exp_id_lime = id_lime.explain_instance(
            x,
            label=1,
            num_samples=2000,
            num_features=d,
            oversample_factor=1.5,
        )
        phi_id_lime = lime_to_vector(exp_id_lime, n_features=d, label=1)
        attributions["ID-LIME"].append(phi_id_lime)

        # SHAP (vanilla)
        shap_vals = shap_explainer.shap_values(x.reshape(1, -1))
        phi_shap = shap_to_vector(shap_vals, d, positive_class=1)
        attributions["SHAP"].append(phi_shap)

        # ID-KernelSHAP
        shap_vals_id = id_shap.explain_instance(
            x,
            nsamples=300,
            n_background=80,
        )
        phi_id_shap = shap_to_vector(shap_vals_id, d, positive_class=1)
        attributions["ID-KernelSHAP"].append(phi_id_shap)

        # Rank consistency per sample
        rank_lime_pairs.append(rank_consistency(phi_lime, phi_id_lime))
        rank_shap_pairs.append(rank_consistency(phi_shap, phi_id_shap))

    # Convert lists -> arrays
    for k in attributions:
        attributions[k] = np.vstack(attributions[k])

    # -----------------------
    # Rank consistency stats
    # -----------------------
    mean_lime = float(np.mean(rank_lime_pairs))
    std_lime = float(np.std(rank_lime_pairs))
    mean_shap = float(np.mean(rank_shap_pairs))
    std_shap = float(np.std(rank_shap_pairs))

    print(f"[INFO] Rank consistency (LIME vs ID-LIME): mean={mean_lime:.3f}, std={std_lime:.3f}")
    print(f"[INFO] Rank consistency (SHAP vs ID-KernelSHAP): mean={mean_shap:.3f}, std={std_shap:.3f}")

    # -----------------------
    # Infidelity computation
    # -----------------------
    print("[INFO] Computing infidelity metrics...")

    infid_results = {
        "LIME": [],
        "ID-LIME": [],
        "SHAP": [],
        "ID-KernelSHAP": [],
    }

    for i in range(N_eval):
        x = X_eval[i]

        phi_lime = attributions["LIME"][i]
        phi_id_lime = attributions["ID-LIME"][i]
        phi_shap = attributions["SHAP"][i]
        phi_id_shap = attributions["ID-KernelSHAP"][i]

        infid_results["LIME"].append(
            compute_infidelity(base_model, x, phi_lime, n_mc=30, noise_scale=0.1, class_idx=1)
        )
        infid_results["ID-LIME"].append(
            compute_infidelity(base_model, x, phi_id_lime, n_mc=30, noise_scale=0.1, class_idx=1)
        )
        infid_results["SHAP"].append(
            compute_infidelity(base_model, x, phi_shap, n_mc=30, noise_scale=0.1, class_idx=1)
        )
        infid_results["ID-KernelSHAP"].append(
            compute_infidelity(base_model, x, phi_id_shap, n_mc=30, noise_scale=0.1, class_idx=1)
        )

    for k in infid_results:
        infid_results[k] = np.array(infid_results[k], dtype=float)

    infid_means = {k: float(v.mean()) for k, v in infid_results.items()}
    infid_stds = {k: float(v.std()) for k, v in infid_results.items()}

    print("[INFO] Infidelity (lower is better):")
    for k in ["LIME", "ID-LIME", "SHAP", "ID-KernelSHAP"]:
        print(f"  {k}: mean={infid_means[k]:.5f}, std={infid_stds[k]:.5f}")

    # -----------------------
    # ROAR & KAR evaluation
    # -----------------------
    fractions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    roar_results = {}
    kar_results = {}

    for method_name, attrib_mat in attributions.items():
        print(f"[INFO] ROAR/KAR for {method_name}...")
        ranking = compute_global_ranking(attrib_mat)
        roar_accs, kar_accs = roar_kar_curves(
            X_train_enc, y_train, X_test_enc, y_test, ranking, fractions
        )
        roar_results[method_name] = roar_accs
        kar_results[method_name] = kar_accs

    # -----------------------
    # SAVE FIGURES TO DISK
    # -----------------------

    # Rank consistency bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["LIME vs ID-LIME", "SHAP vs ID-KernelSHAP"]
    means = [mean_lime, mean_shap]
    errs = [std_lime, std_shap]
    ax.bar(labels, means, yerr=errs, capsize=5)
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Rank Consistency between Vanilla and ID Explainability")
    plt.xticks(rotation=10)
    plt.tight_layout()
    fig.savefig("/content/Capstone-P59/rank_consistency.png", dpi=300)
    plt.close(fig)

    # Infidelity boxplot
    fig_inf, ax_inf = plt.subplots(figsize=(7, 4))
    methods = ["LIME", "ID-LIME", "SHAP", "ID-KernelSHAP"]
    data = [infid_results[m] for m in methods]

    ax_inf.boxplot(
        data,
        tick_labels=methods,
        showmeans=True,
        meanline=False,
        showfliers=False,
    )

    ax_inf.set_ylabel("Infidelity (lower is better)")
    ax_inf.set_title("Infidelity of Explanations on Adult (Logistic Regression)")
    ax_inf.set_yscale("log")
    plt.xticks(rotation=10)
    plt.tight_layout()
    fig_inf.savefig("/content/Capstone-P59/infidelity_boxplot.png", dpi=300)
    plt.close(fig_inf)

    # ROAR plot
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    frac_pct = fractions * 100.0
    for method_name, accs in roar_results.items():
        ax2.plot(frac_pct, accs, marker="o", label=method_name)
    ax2.axhline(base_acc, color="black", linestyle="--", linewidth=1, label="Base (0% removed)")
    ax2.set_xlabel("Fraction of most-important features removed (%)")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("ROAR: Remove And Retrain")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig("/content/Capstone-P59/roar_curves.png", dpi=300)
    plt.close(fig2)

    # KAR plot
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for method_name, accs in kar_results.items():
        ax3.plot(frac_pct, accs, marker="o", label=method_name)
    ax3.axhline(base_acc, color="black", linestyle="--", linewidth=1, label="Base (all features)")
    ax3.set_xlabel("Fraction of most-important features kept (%)")
    ax3.set_ylabel("Test accuracy")
    ax3.set_title("KAR: Keep And Retrain")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig("/content/Capstone-P59/kar_curves.png", dpi=300)
    plt.close(fig3)

    # 3D PCA on-manifold vs off-manifold visualization
    X_off = generate_off_manifold_samples(X_train_enc, n_samples=2000, random_state=0)
    visualize_manifold_vs_off_3d(
        X_train_enc,
        X_off,
        n_points=2000,
        save_path="/content/Capstone-P59/pca_on_vs_off_3d.png",
    )


if __name__ == "__main__":
    main()
