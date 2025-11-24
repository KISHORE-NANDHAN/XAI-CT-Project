#!/usr/bin/env python3
"""
Master aggregator for FEATURE-BASED M9 metrics.

Runs:
  m9/feature_comprehensiveness.py
  m9/feature_sufficiency.py
  m9/feature_insertion_deletion.py

Then aggregates into:
  outputs/m9/m9_feature_summary.json
  outputs/m9/m9_feature_overview.png
  outputs/m9/m9_feature_report.txt
"""

import os, sys, json, argparse, subprocess, logging, yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_config(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

def safe_json(path):
    return json.load(open(path)) if Path(path).exists() else None

# --------------------------
def run(script, config):
    if not Path(script).exists():
        logging.error("Script not found: %s", script)
        return
    cmd = f"python {script} --config {config}"
    logging.info("Running: %s", cmd)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        logging.error("Script failed: %s", script)
    else:
        logging.info("Completed: %s", script)

# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m9.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    outdir = cfg.get("out_dir","outputs/m9")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Run scripts
    run("m9/features/feature_comprehensiveness.py", args.config)
    run("m9/features/feature_sufficiency.py", args.config)
    run("m9/features/feature_insertion_deletion.py", args.config)

    # Load summaries
    comp = safe_json(f"{outdir}/feature_comprehensiveness_summary.json")
    suff = safe_json(f"{outdir}/feature_sufficiency_summary.json")
    insdel = safe_json(f"{outdir}/feature_insertion_deletion_summary.json")

    summary = {
        "comprehensiveness": comp,
        "sufficiency": suff,
        "insertion_deletion": insdel
    }

    # Save master summary
    with open(f"{outdir}/m9_feature_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    # Plot overview
    labels=[]
    vals=[]

    if comp:
        labels.append("Comp Drop")
        vals.append(comp["mean_drop"])
    if suff:
        labels.append("Suff Retained")
        vals.append(suff["mean_retained"])
    if insdel:
        labels.append("AUC Insert")
        vals.append(insdel["mean_insertion_auc"])
        labels.append("AUC Delete")
        vals.append(insdel["mean_deletion_auc"])

    if vals:
        plt.figure(figsize=(8,4))
        plt.bar(labels, vals)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"{outdir}/m9_feature_overview.png", dpi=150)
        plt.close()

    # Text report
    rep_path = f"{outdir}/m9_feature_report.txt"
    with open(rep_path,"w") as f:
        f.write("Feature-Based M9 Report\n\n")
        if comp:
            f.write(f"[Comprehensiveness] mean_drop={comp['mean_drop']:.4f}, n={comp['n']}\n")
        if suff:
            f.write(f"[Sufficiency] mean_retained={suff['mean_retained']:.4f}, n={suff['n']}\n")
        if insdel:
            f.write(f"[Insertion/Deletion] mean_AUC_insert={insdel['mean_insertion_auc']:.4f}, mean_AUC_del={insdel['mean_deletion_auc']:.4f}, n={insdel['n']}\n")

    logging.info("Done. Summary saved to %s", outdir)

if __name__ == "__main__":
    main()

