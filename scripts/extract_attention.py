"""
Given a checkpoint and a dataloader, run the aggregator and export top-k attention slice indices with scores.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from models.mil_transformer import build_aggregator


def load_cfg(path: Path):
    import yaml
    return yaml.safe_load(path.read_text())


def main(cfg_path, ckpt, data_root, out_file, device='cuda'):
    cfg = load_cfg(Path(cfg_path))
    agg = build_aggregator(cfg['aggregator'])
    agg.load_state_dict(torch.load(ckpt, map_location='cpu'))
    agg.to(device).eval()

    # import dataloader
    from scripts.prepare_mil_data import StudyEmbeddingDataset, collate_fn
    from torch.utils.data import DataLoader

    ds = StudyEmbeddingDataset(
        data_root,
        max_slices=cfg['aggregator'].get('max_slices', 256)
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.get('inference_batch', 8),
        collate_fn=collate_fn
    )

    results = []
    with torch.no_grad():
        for embs, labels, uids, slc_inds in tqdm(dl, desc='Inf'):
            embs = embs.to(device)
            logits, attn = agg(embs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            attn = attn.cpu().numpy()  # [B, N]

            for i, uid in enumerate(uids):
                topk = int(cfg.get('top_k', 5))
                idxs = attn[i].argsort()[::-1][:topk].tolist()
                scores = attn[i][idxs].tolist()
                results.append({
                    'study_uid': uid,
                    'label': int(labels[i].item()),
                    'probs': probs[i].tolist(),
                    'topk_indices': idxs,
                    'topk_scores': scores
                })

    Path(out_file).write_text(json.dumps(results, indent=2))
    print('Saved', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Transformer aggregator and export top-k attention slices")
    parser.add_argument('--cfg', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pt)')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--out_file', type=str, required=True, help='Path to save JSON results')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device (cpu or cuda)')
    args = parser.parse_args()

    main(args.cfg, args.ckpt, args.data_root, args.out_file, args.device)
