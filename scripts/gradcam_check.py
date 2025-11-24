# ============================================================
# File: scripts/gradcam_check.py
# Description: Grad-CAM sanity check for trained slice classifier (M3)
# ============================================================
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2, random, json
from PIL import Image
from models.slice_model import get_model
from main_train_slice import load_config


# ------------------------------------------------------------
# Grad-CAM Hook
# ------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        logits, _ = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        loss = logits[:, class_idx].sum()
        loss.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, logits.softmax(1).detach().cpu().numpy()


# ------------------------------------------------------------
# Overlay Utility
# ------------------------------------------------------------
def overlay_cam_on_image(img_tensor, cam, out_path, pred_label, conf):
    img = img_tensor.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.uint8(255 * img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(np.stack([img]*3, axis=-1), 0.5, heatmap, 0.5, 0)

    # Label overlay
    label_text = f"{pred_label} ({conf*100:.1f}%)"
    cv2.putText(blended, label_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    Image.fromarray(blended).save(out_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    cfg = load_config("config/slice_classifier.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🚀 Device: {device}")
    model = get_model(cfg["model"]).to(device)

    # ✅ Safe model load
    ckpt = os.path.join(cfg["output"]["checkpoints_dir"], "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # Target layer selection
    if cfg["model"]["name"].lower() == "resnet50":
        target_layer = model.backbone.layer4[-1]
    elif cfg["model"]["name"].lower() == "convnext_tiny":
        target_layer = model.backbone.features[-1]
    else:
        raise ValueError("Unsupported model for Grad-CAM check")

    gradcam = GradCAM(model, target_layer)

    transform = T.Compose([
        T.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    out_dir = os.path.join(cfg["output"]["logs_dir"], "gradcam_samples")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load sample validation images from manifests
    # --------------------------------------------------------
    img_paths = []
    if "dataset_roots" in cfg["data"]:
        for root in cfg["data"]["dataset_roots"]:
            manifest = os.path.join(root, "val_manifest.json")
            if os.path.exists(manifest):
                with open(manifest, "r") as f:
                    entries = json.load(f)
                    for e in entries:
                        path = e["file_path"]
                        if os.path.exists(path):
                            img_paths.append(path)
    elif "val_dir" in cfg["data"]:
        for label in ["COVID", "NonCOVID"]:
            label_path = os.path.join(cfg["data"]["val_dir"], label)
            if os.path.exists(label_path):
                img_paths += [os.path.join(label_path, f) for f in os.listdir(label_path)]

    print(f"🩻 Found {len(img_paths)} validation slices for Grad-CAM.")

    # --------------------------------------------------------
    # Run Grad-CAM on random subset
    # --------------------------------------------------------
    sample_imgs = random.sample(img_paths, min(5, len(img_paths)))
    labels = ["NonCOVID", "COVID"]

    for img_path in sample_imgs:
        img = Image.open(img_path).convert("L")
        tensor = transform(img).unsqueeze(0).to(device)

        cam, probs = gradcam(tensor)
        pred_idx = probs.argmax(1)[0]
        conf = probs[0][pred_idx]
        pred_label = labels[pred_idx]

        save_name = os.path.basename(img_path).replace(".png", f"_{pred_label}_gradcam.png")
        overlay_cam_on_image(tensor, cam, os.path.join(out_dir, save_name), pred_label, conf)

        print(f"✅ Saved Grad-CAM overlay: {save_name} ({pred_label} {conf*100:.2f}%)")

    print(f"\n🎯 Grad-CAM visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
