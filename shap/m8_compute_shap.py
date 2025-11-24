#!/usr/bin/env python3
"""
m8_compute_shap.py

Compute SHAP values for the fusion head using either TreeSHAP (fast, tree models)
or KernelSHAP (model-agnostic, but slower). Handles multiclass outputs and saves:

    outputs/m8/shap/
        - shap_values.npy              (n_samples x n_features  OR list per-class)
        - shap_summary.csv             (mean |abs| SHAP per feature)
        - shap_summary_bar.png
        - shap_beeswarm.png
        - shap_feature_importance_table.json

Features:
    - automatic model type detection (sklearn tree or generic)
    - optional sample_subset to compute SHAP only for subset
    - caching of background samples (from m8_prepare_background.py)
    - configurable KernelSHAP nsamples (for speed / accuracy tradeoff)
    - multi-thread / parallel support via shap SamplingExplainer if available

Usage examples:
    python scripts/m8_compute_shap.py --embeddings outputs/m6/fused_embeddings.npy \
        --manifest outputs/m6/fused_manifest.csv \
        --model outputs/m6/fusion_head.joblib \
        --background outputs/m8/shap/background_samples.npy \
        --out_dir outputs/m8/shap \
        --method auto

Notes:
    - Project brief referenced in headers: /mnt/data/Problem.docx
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import ClassifierMixin

LOG = logging.getLogger("m8_compute_shap")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOG.addHandler(handler)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_model(model_path: Path):
    LOG.info("Loading fusion head model from %s", model_path)
    model = joblib.load(str(model_path))
    LOG.info("Loaded model type: %s", type(model))
    return model


def is_tree_model(model) -> bool:
    """
    Heuristic: scikit-learn tree ensemble models (RandomForest/GradientBoosting/XGBoost wrapped)
    normally implement 'feature_importances_' attribute and are compatible with TreeExplainer.
    """
    if hasattr(model, "feature_importances_"):
        return True
    # sklearn estimators often subclass ClassifierMixin -- but not determinant for tree.
    return False


def load_embeddings(emb_path: Path):
    LOG.info("Loading embeddings from %s", emb_path)
    return np.load(str(emb_path))


def load_manifest(manifest_path: Optional[Path]):
    if manifest_path:
        LOG.info("Loading manifest from %s", manifest_path)
        return pd.read_csv(str(manifest_path))
    return None


def compute_tree_shap(model, X, out_dir: Path, manifest=None, sample_indices=None):
    LOG.info("Using TreeExplainer (fast) for model: %s", type(model))
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    
    # shap_values shape: for binary: array (n_samples, n_features) or list for multiclass
    LOG.info("Running explainer.shap_values on %d samples", X.shape[0])
    shap_values = explainer.shap_values(X)
    _save_shap_outputs(shap_values, X, out_dir, manifest, sample_indices)
    return shap_values
def compute_kernel_shap(model_predict_fn, X, background, out_dir: Path, nsamples: int = 100, manifests=None, sample_indices=None, seed: int = 42):
    """
    Robust SHAP fallback using a surrogate tree regressor + TreeExplainer.

    Why:
      - KernelExplainer can internally use LassoLarsIC (unstable when n_samples < n_features).
      - We train a small RandomForestRegressor to mimic model_predict_fn on background+X,
        then run TreeExplainer on that surrogate. Tree SHAP is deterministic, fast, and avoids
        the LARS/IC issue.

    Outputs: same as before (saved by _save_shap_outputs).
    """
    from sklearn.ensemble import RandomForestRegressor
    LOG.info("Building surrogate RandomForestRegressor to approximate model outputs (stable TreeSHAP path).")

    # Prepare training data for surrogate: combine background and a subset of X for fidelity.
    # Use up to 500 combined samples to keep surrogate training quick.
    max_surrogate_samples = 500
    bg = np.asarray(background)
    X_all = X
    # choose some samples from X to augment background (improves fidelity)
    n_from_X = min(X_all.shape[0], max(0, max_surrogate_samples - bg.shape[0]))
    if n_from_X > 0:
        # pick uniformly spaced or first-n
        idxs = np.linspace(0, X_all.shape[0] - 1, n_from_X, dtype=int)
        X_train = np.vstack([bg, X_all[idxs]])
    else:
        X_train = bg

    LOG.info("Surrogate training samples: %d (background %d + from-X %d)", X_train.shape[0], bg.shape[0], max(0, n_from_X))

    # Compute target values: use model_predict_fn -> prefer class 1 probability for binary, otherwise array for regressor
    try:
        y_train_raw = model_predict_fn(X_train)
    except Exception as e:
        LOG.exception("model_predict_fn failed on X_train: %s", e)
        raise

    # If predict returns probabilities for multiclass, reduce to single scalar with class-of-interest:
    if isinstance(y_train_raw, np.ndarray) and y_train_raw.ndim == 2:
        # If binary, take column 1; else take argmax-prob as soft-one-hot? use probability of predicted class.
        if y_train_raw.shape[1] == 2:
            y_train = y_train_raw[:, 1]
            LOG.info("Surrogate target: using binary class-1 probability.")
        else:
            # For multiclass, train one surrogate per class would be ideal.
            # Here we explain the first non-background class probability (class 1) as default.
            y_train = y_train_raw[:, 1]
            LOG.info("Surrogate target: multiclass detected, using class index 1 probability as target (change if needed).")
    else:
        # model_predict returned 1D array (probabilities or scalar) — use directly
        y_train = np.asarray(y_train_raw).ravel()

    # Train surrogate regressor
    try:
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        LOG.info("Trained RandomForest surrogate (n_estimators=200, max_depth=8).")
    except Exception as e:
        LOG.exception("Failed to train surrogate RandomForest: %s", e)
        # fallback: try smaller forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        LOG.warning("Trained fallback RandomForest surrogate.")

    # Now explain X with TreeExplainer
    try:
        LOG.info("Running shap.TreeExplainer on surrogate.")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        # shap_values for regressor: (n_samples, n_features); for multiclass it may be list
        _save_shap_outputs(shap_values, X, out_dir, manifests, sample_indices)
        return shap_values
    except Exception as e:
        LOG.exception("TreeExplainer failed on surrogate: %s", e)
        LOG.info("Falling back to shap.Explainer with masker (sampling/permute).")

    # FINAL FALLBACK: shap.Explainer with Independent masker (permutation-based / sampling)
    try:
        # create masker from background
        masker = shap.maskers.Independent(background)
        expl = shap.Explainer(model_predict_fn, masker, algorithm="permutation")
        LOG.info("Running shap.Explainer(..., algorithm='permutation') — this may be slower.")
        shap_values = expl(X, max_evals=nsamples)  # returns Explanation object
        # Convert Explanation to values array if needed
        values = getattr(shap_values, "values", None)
        if values is None:
            # Unexpected — save Explanation object directly
            np.save(str(out_dir / "shap_values_explanation.obj.npy"), shap_values, allow_pickle=True)
            LOG.info("Saved shap Explanation object (fallback).")
        else:
            np.save(str(out_dir / "shap_values.npy"), values, allow_pickle=True)
            _save_shap_outputs(values, X, out_dir, manifests, sample_indices)
        return shap_values
    except Exception as e:
        LOG.exception("All SHAP fallback attempts failed: %s", e)
        raise RuntimeError("Unable to compute SHAP values by surrogate or permutation fallback.") from e


def _save_shap_outputs(shap_values, X, out_dir: Path, manifest, sample_indices):
    ensure_dir(out_dir)
    # Normalize shap_values into (n_samples, n_features) or list per class
    LOG.info("Saving raw shap_values to %s", out_dir / "shap_values.npy")
    np.save(str(out_dir / "shap_values.npy"), shap_values, allow_pickle=True)

    # Compute mean absolute importance across samples and (for multiclass) across classes
    LOG.info("Aggregating mean absolute SHAP values")
    if isinstance(shap_values, list):
        # multiclass: shap_values is list with len=n_classes, each (n_samples, n_features)
        mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
        # aggregate by average across classes
        mean_abs = np.mean(np.vstack(mean_abs_per_class), axis=0)
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    # Save CSV table
    feature_ids = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame({"feature": feature_ids, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.to_csv(str(out_dir / "shap_summary.csv"), index=False)
    LOG.info("Saved shap_summary.csv (top feature: %s)", df.iloc[0]["feature"])

    # Save json table
    df_top = df.head(200)
    df_top.to_json(str(out_dir / "shap_feature_importance_table.json"), orient="records")

    # Generate summary bar and beeswarm plots for top features
    try:
        LOG.info("Generating SHAP plots (bar + beeswarm)")
        # Bar plot (top 30)
        top_k = min(30, X.shape[1])
        top_features_idx = df.head(top_k).index.tolist()
        # For plotting we need shap._explanation objects; create a simple helper
        # If shap_values is list, take the class-mean explanation for plotting
        if isinstance(shap_values, list):
            # average shap for plotting purpose: shape -> (n_samples, n_features)
            shap_mean = np.mean(np.stack(shap_values, axis=0), axis=0)
            shap_v_for_plot = shap.Explanation(values=shap_mean, feature_names=feature_ids)
        else:
            shap_v_for_plot = shap.Explanation(values=shap_values, feature_names=feature_ids)

        plt.tight_layout()
        # Bar summary plot
        try:
            ax = shap.plots.bar(shap_v_for_plot, max_display=top_k, show=False)
            fig = plt.gcf()
            fig.savefig(str(out_dir / "shap_summary_bar.png"), bbox_inches="tight", dpi=150)
            plt.clf()
        except Exception as e:
            LOG.warning("shap.plots.bar failed: %s -- falling back to matplotlib bar", e)
            top_vals = mean_abs[top_features_idx]
            plt.bar(range(len(top_vals)), top_vals)
            plt.xticks(range(len(top_vals)), [feature_ids[i] for i in top_features_idx], rotation=90)
            plt.tight_layout()
            plt.savefig(str(out_dir / "shap_summary_bar.png"), bbox_inches="tight", dpi=150)
            plt.clf()

        # Beeswarm (use shap's beeswarm; if fails, skip)
        try:
            shap.plots.beeswarm(shap_v_for_plot, max_display=top_k, show=False)
            fig = plt.gcf()
            fig.savefig(str(out_dir / "shap_beeswarm.png"), bbox_inches="tight", dpi=150)
            plt.clf()
        except Exception as e:
            LOG.warning("shap.plots.beeswarm failed: %s - skipping beeswarm", e)

    except Exception as e:
        LOG.exception("Failed to generate plots: %s", e)

    LOG.info("Saved SHAP outputs to %s", out_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Compute SHAP values for fusion head (M8)")
    p.add_argument("--embeddings", type=str, required=True, help="Path to fused_embeddings.npy")
    p.add_argument("--manifest", type=str, required=False, help="Path to fused_manifest.csv")
    p.add_argument("--model", type=str, required=True, help="Path to fusion_head.joblib")
    p.add_argument("--background", type=str, required=False, help="Path to background_samples.npy (if omitted, you must supply small background or use TreeSHAP)")
    p.add_argument("--out_dir", type=str, default="outputs/m8/shap", help="Output directory")
    p.add_argument("--method", type=str, choices=["auto", "tree", "kernel"], default="auto",
                   help="Explainer method. 'auto' chooses tree if model appears tree-based.")
    p.add_argument("--nsamples", type=int, default=200, help="nsamples for KernelSHAP (higher -> slower more accurate)")
    p.add_argument("--subset", type=int, default=None, help="If set, compute SHAP only for first N samples (useful for speed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    LOG.info("m8_compute_shap started. Project brief: /mnt/data/Problem.docx")

    # Load inputs
    X = load_embeddings(Path(args.embeddings))
    manifest = load_manifest(Path(args.manifest)) if args.manifest else None
    model = load_model(Path(args.model))

    # Optionally reduce to subset for SHAP compute
    if args.subset is not None:
        N = min(args.subset, X.shape[0])
        LOG.info("Using subset: first %d samples for SHAP computation", N)
        X_use = X[:N]
        sample_indices = list(range(N))
    else:
        X_use = X
        sample_indices = None

    # Decide explainer
    chosen_method = args.method
    if args.method == "auto":
        try:
            if is_tree_model(model):
                chosen_method = "tree"
            else:
                chosen_method = "kernel"
        except Exception:
            chosen_method = "kernel"
    LOG.info("Chosen explainer method: %s", chosen_method)

    if chosen_method == "tree":
        shap_values = compute_tree_shap(model, X_use, out_dir, manifest, sample_indices)
    elif chosen_method == "kernel":
        if not args.background:
            raise ValueError("KernelSHAP requires a background dataset. Provide --background path or run m8_prepare_background.py")
        background = np.load(str(args.background))
        # build prediction wrapper: expects function returning probability vector or single output
        def model_predict(Xinput):
            try:
                proba = model.predict_proba(Xinput)
                # If multiclass, shap expects either single-output or handle list shape; keep as-is
                if proba.shape[1] == 2:
                    return proba[:, 1]
                return proba
            except Exception:
                # fallback to direct predict
                return model.predict(Xinput)

        shap_values = compute_kernel_shap(model_predict, X_use, background, out_dir, nsamples=args.nsamples, manifests=manifest, sample_indices=sample_indices)
    else:
        raise ValueError("Unknown method")

    LOG.info("SHAP computation finished. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
