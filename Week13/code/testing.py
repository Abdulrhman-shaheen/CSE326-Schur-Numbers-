# Loads the model and try to test it. A simple interfacec.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from SchurTransfomer import load_model, SchurTransformer, verify_transformer
import random
from collections import defaultdict
import matplotlib.pyplot as plt


model = SchurTransformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(model, "data/schur_transformer.pt")



test_seq = []
length_stats = defaultdict(list)

for i in range(7, 70):
    file_path = f"data/pairs_z{i}.txt"
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            sampled = random.sample(lines, min(500, len(lines)))
            seqs = [eval(line.strip()) for line in sampled]
            test_seq.extend(seqs)
    except FileNotFoundError:
        print(f"File {file_path} not found, skipping.")

for seq in test_seq:
    # Copy to avoid modifying original
    seq = seq.copy()
    valid = True
    input_len = len(seq)
    z = seq[-1][0] + 1

    while valid:
        c, valid = verify_transformer(model, seq, z)
        seq.append((z, c))
        z += 1

    output_len = len(seq)
    length_stats[input_len].append(output_len)

# Compute averages
input_lengths = sorted(length_stats.keys())
avg_output_lengths = [sum(length_stats[l]) / len(length_stats[l]) for l in input_lengths]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(input_lengths, avg_output_lengths, marker='o')
plt.xlabel("Input Sequence Length")
plt.ylabel("Average Output Sequence Length")
plt.title("Growth of Sequence Length vs. Input Length")
plt.grid(True)
plt.savefig("sequence_growth_plot.png")
plt.show()