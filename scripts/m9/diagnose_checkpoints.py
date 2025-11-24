# scripts/m9/diagnose_checkpoints.py
import torch, joblib, json, sys, os
from pathlib import Path

def inspect_torch(path):
    print(">>> Inspecting:", path)
    try:
        obj = torch.load(path, map_location='cpu')
        print(" type:", type(obj))
        if hasattr(obj, 'state_dict'):
            print("  Looks like nn.Module instance.")
        elif isinstance(obj, dict):
            print("  It's a dict (likely state_dict or saved dict). Keys:", list(obj.keys())[:10])
        else:
            print("  Other torch object")
    except Exception as e:
        print("  Failed to torch.load:", e)

def inspect_joblib(path):
    print(">>> Inspecting joblib:", path)
    try:
        obj = joblib.load(path)
        print(" type:", type(obj))
        # if sklearn estimator
        try:
            import sklearn
            from sklearn.base import BaseEstimator
            print("  Is sklearn estimator?", isinstance(obj, BaseEstimator))
        except Exception:
            pass
        if hasattr(obj, 'coef_'):
            print("  has coef_, shape:", getattr(obj, 'coef_').shape)
        if hasattr(obj, 'predict_proba'):
            print("  has predict_proba")
    except Exception as e:
        print("  Failed to joblib.load:", e)

if __name__ == '__main__':
    base = Path('.')
    # candidate model files (update if different)
    cand = ["./outputs/checkpoints/best_model.pt",
            "./outputs/m5/checkpoints/best.pth",
            "./models/fusion_model.pt",
            "./outputs/m6/fusion_head.joblib"]
    for p in cand:
        if Path(p).exists():
            if p.endswith(('.pt','.pth')):
                inspect_torch(p)
            elif p.endswith('.joblib') or p.endswith('.pkl'):
                inspect_joblib(p)
        else:
            print("NOT FOUND:", p)
