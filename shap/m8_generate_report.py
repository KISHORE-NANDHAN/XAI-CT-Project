#!/usr/bin/env python3
"""
m8_generate_report.py

Generate M8 PDF report (clinician-facing one-pager + extended multipage report) that
combines:
  - SHAP summary bar + beeswarm (outputs/m8/shap/)
  - Top feature table (shap_summary.csv)
  - Prototype tiles (outputs/m8/prototypes/prototype_tiles/)
  - Calibration diagram & ROC (outputs/m6/)
  - Basic metadata (fused_manifest.csv)

Outputs:
  - outputs/m8/m8_report.pdf
  - outputs/m8/m8_report_extended.pdf  (optional multi-page with larger grids)

Notes:
  - Project brief path (uploaded file): /mnt/data/Problem.docx
  - Requires: reportlab, pillow, numpy, pandas, matplotlib
  - Run after: scripts/m8_prepare_background.py, scripts/m8_compute_shap.py,
               scripts/m8_prototype_extractor.py, scripts/m8_prototype_tiles.py

Example:
    python scripts/m8_generate_report.py --config m8_config.yaml

"""

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet

LOG = logging.getLogger("m8_generate_report")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
LOG.addHandler(handler)

# Path to the uploaded project brief (user-provided)
PROJECT_BRIEF_PATH = "/mnt/data/Problem.docx"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def collect_assets(cfg: dict) -> dict:
    """
    Validate presence of expected assets and return resolved paths.
    """
    assets = {}
    # SHAP folder and files
    shap_dir = Path(cfg["shap"]["out_dir"])
    assets["shap_dir"] = shap_dir
    assets["shap_bar"] = shap_dir / "shap_summary_bar.png"
    assets["shap_beeswarm"] = shap_dir / "shap_beeswarm.png"
    assets["shap_csv"] = shap_dir / "shap_summary.csv"

    # Prototype tiles
    proto_tiles_dir = Path(cfg["prototypes"]["tiles_dir"])
    assets["proto_tiles_dir"] = proto_tiles_dir

    # M6 artifacts: calibration and ROC
    m6_dir = Path(cfg["m6"]["out_dir"])
    assets["calibration"] = m6_dir / "calibration_diagram.png"
    assets["roc"] = m6_dir / "roc_multiclass.png"

    # Manifest
    assets["manifest"] = Path(cfg["m6"]["manifest"])

    # output dir
    assets["out_dir"] = Path(cfg["m8"]["out_dir"])
    ensure_dir(assets["out_dir"])

    # verify existence, but don't fail hard — some are optional
    for k, p in list(assets.items()):
        if isinstance(p, Path) and not p.exists():
            LOG.warning("Expected asset '%s' not found at %s", k, p)
    return assets


def pil_image_for_rl(path: Path, max_w_mm: float = None, max_h_mm: float = None) -> RLImage:
    """
    Load image via PIL and prepare ReportLab Image object with proper scaling.
    max_w_mm/h_mm specify bounding box in millimeters.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    pil = Image.open(path)
    # compute scaling
    w_px, h_px = pil.size
    # convert px to points assuming 72 dpi for ReportLab default; simplest approach:
    # We'll fit by mm conversion: 1 mm = 2.83464567 points
    if max_w_mm or max_h_mm:
        mm_to_pt = 72.0 / 25.4
        max_w_pt = max_w_mm * mm_to_pt if max_w_mm else None
        max_h_pt = max_h_mm * mm_to_pt if max_h_mm else None
        # compute scale
        img_ratio = w_px / h_px
        if max_w_pt and max_h_pt:
            target_w = min(max_w_pt, max_h_pt * img_ratio)
            target_h = target_w / img_ratio
        elif max_w_pt:
            target_w = max_w_pt
            target_h = target_w / img_ratio
        elif max_h_pt:
            target_h = max_h_pt
            target_w = target_h * img_ratio
        else:
            target_w = w_px
            target_h = h_px
        bio = io.BytesIO()
        pil.save(bio, format="PNG")
        bio.seek(0)
        rlimg = RLImage(bio, width=target_w, height=target_h)
        return rlimg
    else:
        bio = io.BytesIO()
        pil.save(bio, format="PNG")
        bio.seek(0)
        return RLImage(bio)


def build_top_feature_table(shap_csv: Path, n_top: int = 10):
    """
    Return a ReportLab Table containing top-n features and their mean abs SHAP value.
    shap_csv expected to have columns: feature, mean_abs_shap
    """
    if not shap_csv.exists():
        LOG.warning("shap summary csv not found: %s", shap_csv)
        return None
    df = pd.read_csv(shap_csv)
    df_top = df.head(n_top)
    data = [["Rank", "Feature", "Mean |SHAP|"]]
    for i, row in df_top.iterrows():
        data.append([i + 1, str(row["feature"]), f"{float(row['mean_abs_shap']):.5f}"])
    table = Table(data, colWidths=[30 * mm, 70 * mm, 40 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    return table


def gather_proto_tiles(tiles_dir: Path, max_tiles: int = 12) -> List[Path]:
    """
    Collect up-to max_tiles prototype tile images sorted by name.
    """
    if not tiles_dir.exists():
        LOG.warning("Prototype tiles dir does not exist: %s", tiles_dir)
        return []
    imgs = sorted([p for p in tiles_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    return imgs[:max_tiles]


def create_pdf_report(assets: dict, cfg: dict):
    """
    Compose a two-page PDF:
    - page 1: Title, SHAP bar + top features + calibration + ROC
    - page 2: Prototype tile grid + small manifest stats
    """
    out_pdf = assets["out_dir"] / "m8_report.pdf"
    LOG.info("Creating report at %s", out_pdf)
    doc = BaseDocTemplate(str(out_pdf), pagesize=portrait(A4), leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("XAI-CT — M8 Explainability Report (SHAP & Prototypes)", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 6 * mm))

    # short metadata and project brief reference
    meta_txt = f"Project brief: {PROJECT_BRIEF_PATH}  —  Module: M8 Feature Attribution & Prototypes"
    story.append(Paragraph(meta_txt, styles["Normal"]))
    story.append(Spacer(1, 4 * mm))

    # Layout: left column SHAP bar + beeswarm (stacked), right column table/calib/roc
    # We'll insert images with scaling
    shap_bar = assets.get("shap_bar")
    shap_beeswarm = assets.get("shap_beeswarm")
    shap_table = build_top_feature_table(assets.get("shap_csv"), n_top=8)

    # Row: SHAP bar (full width)
    if shap_bar and shap_bar.exists():
        try:
            img = pil_image_for_rl(shap_bar, max_w_mm=170, max_h_mm=60)
            story.append(img)
            story.append(Spacer(1, 4 * mm))
        except Exception as e:
            LOG.warning("Failed to add SHAP bar image: %s", e)

    # Add beeswarm if exists
    if shap_beeswarm and shap_beeswarm.exists():
        try:
            img = pil_image_for_rl(shap_beeswarm, max_w_mm=170, max_h_mm=60)
            story.append(img)
            story.append(Spacer(1, 4 * mm))
        except Exception as e:
            LOG.warning("Failed to add SHAP beeswarm: %s", e)

    # Add top features table
    if shap_table:
        story.append(Paragraph("Top SHAP features (mean |SHAP|)", styles["Heading3"]))
        story.append(spacer := Spacer(1, 2 * mm))
        story.append(shap_table)
        story.append(Spacer(1, 6 * mm))

    # Calibration + ROC
    story.append(Paragraph("Calibration & ROC", styles["Heading3"]))
    calib_img = assets.get("calibration")
    roc_img = assets.get("roc")
    if calib_img and calib_img.exists():
        try:
            story.append(pil_image_for_rl(calib_img, max_w_mm=80, max_h_mm=60))
        except Exception as e:
            LOG.warning("Failed to add calibration image: %s", e)
    if roc_img and roc_img.exists():
        try:
            story.append(pil_image_for_rl(roc_img, max_w_mm=80, max_h_mm=60))
        except Exception as e:
            LOG.warning("Failed to add ROC image: %s", e)

    story.append(Spacer(1, 6 * mm))
    # Page break manually by adding large spacer and then second page content
    story.append(Paragraph("Prototype Gallery", styles["Heading2"]))
    story.append(Spacer(1, 4 * mm))

    # Prototype grid: 3 columns
    proto_imgs = gather_proto_tiles(assets.get("proto_tiles_dir"), max_tiles=12)
    if proto_imgs:
        # We'll build a table grid where each cell is an image
        cells_per_row = 3
        grid = []
        row = []
        cell_w_mm = (180 / cells_per_row)  # approx available width
        for i, p in enumerate(proto_imgs):
            try:
                rlimg = pil_image_for_rl(p, max_w_mm=cell_w_mm, max_h_mm=cell_w_mm)
                row.append(rlimg)
            except Exception as e:
                LOG.warning("Failed to prepare prototype image %s: %s", p, e)
                row.append(Paragraph(str(p.name), styles["Normal"]))
            if (i + 1) % cells_per_row == 0:
                grid.append(row)
                row = []
        if row:
            # pad
            while len(row) < cells_per_row:
                row.append(Paragraph("", styles["Normal"]))
            grid.append(row)
        # Convert grid into Table
        tbl = Table(grid, colWidths=[60 * mm] * cells_per_row)
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(tbl)
    else:
        story.append(Paragraph("No prototype tiles found.", styles["Normal"]))

    # small manifest stats
    story.append(Spacer(1, 6 * mm))
    try:
        manifest = pd.read_csv(assets["manifest"])
        n_studies = len(manifest)
        n_sites = manifest["site"].nunique() if "site" in manifest.columns else "N/A"
        stats = f"Manifest: {assets['manifest']}  —  #rows: {n_studies}  —  #sites: {n_sites}"
        story.append(Paragraph(stats, styles["Normal"]))
    except Exception as e:
        LOG.warning("Failed to read manifest for stats: %s", e)

    # Build doc
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    template = PageTemplate(id="OneCol", frames=[frame])
    doc.addPageTemplates([template])
    doc.build(story)
    LOG.info("Saved report: %s", out_pdf)


def main():
    parser = argparse.ArgumentParser(description="Generate M8 report PDF")
    parser.add_argument("--config", type=str, default="m8_config.yaml", help="Path to M8 config yaml")
    parser.add_argument("--out", type=str, default=None, help="Optional override output path for PDF")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    assets = collect_assets(cfg)

    # allow override of output name
    if args.out:
        assets["out_dir"] = Path(args.out)
        ensure_dir(assets["out_dir"])

    create_pdf_report(assets, cfg)
    LOG.info("m8_generate_report.py finished.")


if __name__ == "__main__":
    main()
