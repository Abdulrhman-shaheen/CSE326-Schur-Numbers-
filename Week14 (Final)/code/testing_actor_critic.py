import torch
import random
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- import your Actor–Critic and mask fn ---
from actor_critic_transformer import SchurActorCritic, get_invalid_color_mask

# reproducibility
random.seed(0)
torch.manual_seed(0)

# 1) load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SchurActorCritic()
ckpt = torch.load("schur_actorcritic_rl.pt", map_location=device)
model.load_state_dict(ckpt, strict=False)
model.to(device).eval()

# 2) gather test sequences by input length
test_seqs = defaultdict(list)
for z in range(7, 70):
    path = f"data/pairs_z{z}.txt"
    if not os.path.exists(path):
        continue
    lines = open(path).read().splitlines()
    sampled = random.sample(lines, min(500, len(lines)))
    for ln in sampled:
        seq = ast.literal_eval(ln)
        test_seqs[len(seq)].append(seq)

# 3) run greedy roll-out for each input length
length_stats = defaultdict(list)
for inp_len, seqs in test_seqs.items():
    for base_seq in seqs:
        seq = base_seq.copy()
        z = seq[-1][0] + 1
        # extend until no valid action remains
        while True:
            # build mask + inputs
            mask = get_invalid_color_mask(seq, z).unsqueeze(0).to(device)
            nums = (
                torch.tensor([p[0] for p in seq], dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )
            cols = (
                torch.tensor([p[1] for p in seq], dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                probs, _ = model(nums, cols, mask)
            action = torch.argmax(probs, dim=-1).item()
            color = action + 1

            # if it’s invalid, stop
            if get_invalid_color_mask(seq, z)[color - 1]:
                break

            seq.append((z, color))
            z += 1

        length_stats[inp_len].append(len(seq))

# 4) compute averages & plot
input_lens = sorted(length_stats)
avg_outs = [np.mean(length_stats[L]) for L in input_lens]

plt.figure(figsize=(8, 5))
plt.plot(input_lens, avg_outs, marker="o")
plt.xlabel("Input Sequence Length")
plt.ylabel("Avg. Extended Sequence Length")
plt.title("RL Policy: Sequence Growth vs. Input Length")
plt.grid(True)
plt.tight_layout()
plt.savefig("rl_sequence_growth.png")
plt.show()
