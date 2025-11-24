# ============================================================
# File: scripts/save_training_log.py
# Description: Save provided console output as training_log.txt
# ============================================================

import os

# Multi-line string containing your console output
log_text = """
Device: cuda
[INFO] train: classes={'CAP': 0, 'COVID': 1, 'NORMAL': 2, 'PNEUMONIA': 3}
[INFO] val: classes={'CAP': 0, 'COVID': 1, 'NORMAL': 2, 'PNEUMONIA': 3}
[INFO] test: classes={'CAP': 0, 'COVID': 1, 'NORMAL': 2, 'PNEUMONIA': 3}
[INFO] Found 1351 train / 167 val / 172 test volumes
[INFO] Target shape = [1, 96, 160, 160]
torch.compile disabled to avoid Triton error.

Epoch 1/20
Train loss 0.9810 | Train AUC 0.721248015186148
Val loss   4.6182 | Val AUC   0.6905605268760221
✅ Saved new best -> outputs/m5/checkpoints\best.pth

Epoch 2/20
Train loss 0.7086 | Train AUC 0.8217694533676508
Val loss   2.3541 | Val AUC   0.7652750184282291
✅ Saved new best -> outputs/m5/checkpoints\best.pth

Epoch 3/20
Train loss 0.5838 | Train AUC 0.8721621571398176
Val loss   13.4786 | Val AUC   0.5043674787548371

Epoch 4/20
Train loss 0.5581 | Train AUC 0.8670253704819417
Val loss   0.7252 | Val AUC   0.9139049183907297
✅ Saved new best -> outputs/m5/checkpoints\best.pth

Epoch 5/20
Train loss 0.5481 | Train AUC 0.8698928810666148
Val loss   0.6804 | Val AUC   0.9316071488481772
✅ Saved new best -> outputs/m5/checkpoints\best.pth

Epoch 6/20
Train loss 0.5326 | Train AUC 0.8696882490813366
Val loss   1.4931 | Val AUC   0.9178719055882761

Epoch 7/20
Train loss 0.5157 | Train AUC 0.889228917042478
Val loss   0.7549 | Val AUC   0.8957257254886353

Epoch 8/20
Train loss 0.4982 | Train AUC 0.8914171729385381
Val loss   21.7977 | Val AUC   0.5145936735180888

Epoch 9/20
Train loss 0.4848 | Train AUC 0.9052747943427357
Val loss   0.5432 | Val AUC   0.9054390867921795

Epoch 10/20
Train loss 0.4263 | Train AUC 0.9178603263698333
Val loss   0.4970 | Val AUC   0.9102279377092612

Epoch 11/20
Train loss 0.4155 | Train AUC 0.9128652927998915
Val loss   4.6835 | Val AUC   0.8249214703261486

Epoch 12/20
Train loss 0.4013 | Train AUC 0.933463550060109
Val loss   10.2824 | Val AUC   0.7181082607584366

Epoch 13/20
Train loss 0.4015 | Train AUC 0.9406039561787224
Val loss   0.5793 | Val AUC   0.9321516834443049
✅ Saved new best -> outputs/m5/checkpoints\best.pth

Epoch 14/20
Train loss 0.3924 | Train AUC 0.9329626141664541
Val loss   1.0277 | Val AUC   0.8903578307245241

Epoch 15/20
Train loss 0.3916 | Train AUC 0.9352946919288001
Val loss   0.4823 | Val AUC   0.9335064959805582
✅ Saved new best -> outputs/m5/checkpoints\best.pth

Epoch 16/20
Train loss 0.3665 | Train AUC 0.942908008224941
Val loss   0.6015 | Val AUC   0.8878568518207388

Epoch 17/20
Train loss 0.3335 | Train AUC 0.9564287002167806
Val loss   6.3391 | Val AUC   0.8263789016717217

Epoch 18/20
Train loss 0.3663 | Train AUC 0.9512693776220131
Val loss   1.0312 | Val AUC   0.9002817927630377

Epoch 19/20
Train loss 0.3606 | Train AUC 0.9561685725473491
Val loss   0.8389 | Val AUC   0.9105520945702981

Epoch 20/20
Train loss 0.3159 | Train AUC 0.9641405338421081
Val loss   0.7198 | Val AUC   0.9355369309463852
✅ Saved new best -> outputs/m5/checkpoints\best.pth
"""

os.makedirs("outputs/m5", exist_ok=True)
with open("outputs/m5/training_log.txt", "w", encoding="utf-8") as f:
    f.write(log_text.strip())
print("✅ Saved log file → outputs/m5/training_log.txt")
