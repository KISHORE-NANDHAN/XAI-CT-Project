#!/usr/bin/env python3
"""
Generate feature-level visuals for M9.

Reads:
  - outputs/m9/feature_comprehensiveness.json
  - outputs/m9/feature_sufficiency.json
  - outputs/m9/feature_insertion_deletion.json

Writes:
  outputs/m9/visuals/
    - comp_distribution.png
    - suff_distribution.png
    - auc_boxplots.png
    - importance_global_top20.png
    - insertion_deletion_examples/ (per-study curves)
    - per_study_summary/*.png

Usage:
  python m9/feature_visuals.py --config config/m9.yaml
"""
import os
import json
import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # safe backend
plt.rcParams.update({'figure.dpi': 150})

def mkdir(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

def load_json(p):
    p = Path(p)
    if not p.exists():
        print(f"WARNING: {p} not found")
        return None
    with open(p, "r") as f:
        return json.load(f)

def safe_get_study_id(item, idx):
    return item.get("study_id") if isinstance(item, dict) and "study_id" in item else f"study_{idx}"

def plot_hist(values, title, xlabel, outpath, bins=40):
    plt.figure(figsize=(6,3))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_box(values_dict, title, outpath):
    labels = list(values_dict.keys())
    data = [values_dict[k] for k in labels]
    plt.figure(figsize=(6,3))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_bar_simple(labels, vals, title, outpath):
    plt.figure(figsize=(6,3))
    plt.bar(labels, vals)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_insertion_deletion_curves(result_item, outpath_prefix):
    ins = result_item.get("insertion_curve")
    dele = result_item.get("deletion_curve")
    idx = result_item.get("index", 0)
    pred = result_item.get("pred_class", None)
    orig = result_item.get("orig_prob", None)
    if ins is None or dele is None:
        return
    steps = len(ins)
    xs = np.linspace(0, 1, steps)
    # insertion
    plt.figure(figsize=(5,3))
    plt.plot(xs, ins, label="insertion")
    plt.plot(xs, dele, label="deletion")
    plt.xlabel("Fraction of top-k features (0->1)")
    plt.ylabel("Predicted probability")
    plt.title(f"Study {idx} | pred={pred} | orig_prob={orig:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outpath_prefix}_study{idx:03d}.png")
    plt.close()

def plot_global_importance_from_model(model_path, embeddings_path, outpath, topk=20):
    # Attempt to extract input-layer weight importances from TorchScript if possible.
    try:
        import torch
        if Path(model_path).exists():
            ts = torch.jit.load(str(model_path), map_location="cpu")
            # try named_parameters
            try:
                params = {k: v for k, v in ts.named_parameters()}
            except Exception:
                params = {}
            imp = None
            for name, p in params.items():
                arr = p.detach().cpu().numpy()
                if arr.ndim == 2:
                    # weight shape (out, in)
                    imp = np.abs(arr).mean(axis=0)
                    break
            if imp is None:
                raise RuntimeError("no suitable param found")
            idx = np.argsort(imp)[::-1][:topk]
            vals = imp[idx]
            labels = [f"f{int(i)}" for i in idx]
            plt.figure(figsize=(8,3))
            plt.bar(labels, vals)
            plt.title("Global feature importance (top %d)" % topk)
            plt.xticks(rotation=60)
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()
            return True
    except Exception as e:
        print("Could not extract model weights for importance:", e)

    # fallback: importance by embedding variance
    try:
        X = np.load(embeddings_path)
        var = np.var(X, axis=0)
        idx = np.argsort(var)[::-1][:topk]
        vals = var[idx]
        labels = [f"f{int(i)}" for i in idx]
        plt.figure(figsize=(8,3))
        plt.bar(labels, vals)
        plt.title("Global feature importance (var fallback) top %d" % topk)
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return True
    except Exception as e:
        print("importance fallback failed:", e)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m9.yaml")
    args = parser.parse_args()

    # default locations (overridable through config file read below)
    cfg = {}
    if Path(args.config).exists():
        import yaml
        with open(args.config,"r") as f:
            try:
                cfg = yaml.safe_load(f)
            except Exception:
                cfg = {}
    outdir = Path(cfg.get("out_dir", "outputs/m9"))
    visdir = outdir / "visuals"
    mkdir(visdir)

    # file paths
    comp_j = cfg.get("comp_json", str(outdir / "feature_comprehensiveness.json"))
    suff_j = cfg.get("suff_json", str(outdir / "feature_sufficiency.json"))
    insdel_j = cfg.get("insdel_json", str(outdir / "feature_insertion_deletion.json"))
    emb_path = cfg.get("embeddings_path", "outputs/m6/fused_embeddings.npy")
    model_path = cfg.get("model_path", "")

    comp = load_json(comp_j)
    suff = load_json(suff_j)
    insdel = load_json(insdel_j)

    # ------- GLOBAL overview -------
    # bar overview (if m9_feature_summary exists, use it)
    summary_path = outdir / "m9_feature_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        # create simple bar plot
        vals = []
        labels = []
        if summary.get("comprehensiveness"):
            labels.append("Comp Drop")
            vals.append(summary["comprehensiveness"]["mean_drop"])
        if summary.get("sufficiency"):
            labels.append("Suff Retained")
            vals.append(summary["sufficiency"]["mean_retained"])
        if summary.get("insertion_deletion"):
            labels.append("AUC Insert")
            vals.append(summary["insertion_deletion"]["mean_insertion_auc"])
            labels.append("AUC Delete")
            vals.append(summary["insertion_deletion"]["mean_deletion_auc"])
        plot_bar_simple(labels, vals, "M9 Feature Overview", str(visdir / "overview_bar.png"))

    # ------- distributions -------
    if comp and "results" in comp:
        drops = [r.get("prob_drop", 0.0) for r in comp["results"]]
        plot_hist(drops, "Comprehensiveness: prob_drop distribution", "prob_drop", str(visdir / "comp_distribution.png"))
    if suff and "results" in suff:
        retained = [r.get("prob_retained", 0.0) for r in suff["results"]]
        plot_hist(retained, "Sufficiency: prob_retained distribution", "prob_retained", str(visdir / "suff_distribution.png"))

    # ------- insertion/deletion boxplots -------
    if insdel and "results" in insdel:
        ins_vals = [r.get("auc_insertion", 0.0) for r in insdel["results"]]
        del_vals = [r.get("auc_deletion", 0.0) for r in insdel["results"]]
        boxdict = {"Insertion AUC": ins_vals, "Deletion AUC": del_vals}
        plot_box(boxdict, "Insertion/Deletion AUC (per-study)", str(visdir / "auc_boxplots.png"))

    # ------- global importance (top20) -------
    plot_global_importance_from_model(model_path, emb_path, str(visdir / "importance_global_top20.png"), topk=20)

    # ------- per-study insertion/deletion example curves -------
    exdir = visdir / "insertion_deletion_examples"
    mkdir(exdir)
    if insdel and "results" in insdel:
        for idx, item in enumerate(insdel["results"]):
            # plot every Nth study to avoid too many figures (here N=10)
            if idx % 10 == 0:
                try:
                    plot_insertion_deletion_curves(item, str(exdir / "insdel"))
                except Exception as e:
                    print("failed per-study plot", idx, e)

    # ------- per-study summary small cards -------
    summary_dir = visdir / "per_study_summary"
    mkdir(summary_dir)
    if comp and "results" in comp and suff and "results" in suff:
        # assume same ordering
        for i, c in enumerate(comp["results"]):
            s = suff["results"][i] if i < len(suff["results"]) else {}
            id_ = safe_get_study_id(c, i)
            fig, ax = plt.subplots(figsize=(4,2))
            # bar: orig prob, masked prob, kept prob
            orig = c.get("orig_prob", 0.0)
            masked = c.get("masked_prob", 0.0)
            kept = s.get("kept_prob", 0.0)
            ax.bar(["orig","masked","kept"], [orig, masked, kept])
            ax.set_ylim(0,1)
            ax.set_title(f"{id_} idx:{i}")
            plt.tight_layout()
            fpath = summary_dir / f"{id_}_summary.png"
            fig.savefig(str(fpath))
            plt.close(fig)

    print("Saved visuals to", str(visdir))
    # print optional uploaded file path (developer-provided)
    uploaded_doc = "/mnt/data/Problem.docx"
    if Path(uploaded_doc).exists():
        print("Optional uploaded file available at:", uploaded_doc)

if __name__ == "__main__":
    main()
