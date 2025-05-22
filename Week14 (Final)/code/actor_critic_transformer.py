# I Have no idea, hopefully you dont have to as well.

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import ast
import os

# Transformer Model
import torch
import torch.nn as nn


class SchurActorCritic(nn.Module):
    def __init__(
        self, num_vocab=192, color_vocab=5, embed_dim=64, nhead=4, num_layers=2
    ):
        super().__init__()
        # Embeddings
        self.num_embed = nn.Embedding(num_vocab + 1, embed_dim // 2)
        self.color_embed = nn.Embedding(color_vocab + 1, embed_dim // 2)
        # Learned positional encodings
        self.pos_encoder = nn.Parameter(torch.randn(1, 200, embed_dim))
        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Actor (policy) head
        self.policy_head = nn.Linear(embed_dim, color_vocab)
        # Critic (value) head
        self.value_head = nn.Linear(embed_dim, 1)
        # Optional softmax for policy output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, numbers, colors, mask=None):
        """
        numbers: LongTensor (B, L)
        colors:  LongTensor (B, L)
        mask:    BoolTensor (B, C) where True = invalid action
        returns: probs (B, C), value (B,)
        """
        # Embed and combine
        num_emb = self.num_embed(numbers)  # (B, L, D/2)
        color_emb = self.color_embed(colors)  # (B, L, D/2)
        x = torch.cat([num_emb, color_emb], dim=-1)  # (B, L, D)
        # Add pos encoding
        x = x + self.pos_encoder[:, : x.size(1), :]  # (B, L, D)
        # Transformer
        x = self.transformer(x)  # (B, L, D)
        last = x[:, -1, :]  # (B, D)

        # Policy logits & masking
        logits = self.policy_head(last)  # (B, C)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)  # instead of -inf
        probs = torch.softmax(logits, dim=-1)  # will now be valid

        # State-value
        value = self.value_head(last).squeeze(-1)  # (B,)

        return probs, value


# Dataset
class SchurDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        numbers = [pair[0] for pair in seq]
        colors = [pair[1] for pair in seq]
        target = colors[-1] if len(seq) > 1 else colors[0]
        return (
            torch.tensor(numbers[:-1], dtype=torch.long),
            torch.tensor(colors[:-1], dtype=torch.long),
            torch.tensor(target - 1, dtype=torch.long),
        )


# Constraint Masking
def get_invalid_color_mask(pairs, z):
    mask = torch.zeros(5, dtype=torch.bool)
    color_nums = {i: [] for i in range(1, 6)}
    for num, color in pairs:
        color_nums[color].append(num)
    for x in range(1, z // 2 + 1):
        y = z - x
        for c in range(1, 6):
            if x in color_nums[c] and y in color_nums[c]:
                mask[c - 1] = True
    return mask


# Data Loading
def load_data(data_dir):
    sequences = []
    for z in range(6, 69):
        file_path = os.path.join(data_dir, f"pairs_z{z}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    seq = ast.literal_eval(line.strip())
                    sequences.append(seq)
    return sequences


# Training Loop
def train_transformer(model, dataloader, epochs=10, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for numbers, colors, target in dataloader:
            numbers, colors, target = (
                numbers.to(device),
                colors.to(device),
                target.to(device),
            )
            optimizer.zero_grad()
            output = model(numbers, colors)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
    return model


# Load Model
def load_model(model, path="schur_transformer.pt", device="cuda"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model


# Verification
def verify_transformer(model, test_seq, z, device="cuda"):
    model.eval()
    pairs = test_seq[:-1]
    numbers = (
        torch.tensor([p[0] for p in pairs], dtype=torch.long).unsqueeze(0).to(device)
    )
    colors = (
        torch.tensor([p[1] for p in pairs], dtype=torch.long).unsqueeze(0).to(device)
    )
    mask = get_invalid_color_mask(pairs, z).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(numbers, colors, mask)
    predicted_color = torch.argmax(probs, dim=1).item() + 1
    color_nums = [p[0] for p in pairs if p[1] == predicted_color]
    valid = all(
        x not in color_nums or (z - x) not in color_nums for x in range(1, z // 2 + 1)
    )
    print(f"z={z}, Predicted color: {predicted_color}, Valid: {valid}")
    return predicted_color, valid


if __name__ == "__main__":
    # Load data
    data_dir = "/home/abdelrahman/schur-rl/data/"  # Replace with actual path
    sequences = load_data(data_dir)
    print(f"Loaded {len(sequences)} sequences.")
    dataset = SchurDataset(sequences)

    def collate_fn(batch):
        numbers, colors, targets = zip(*batch)
        max_len = max(len(n) for n in numbers)
        padded_numbers = torch.zeros(len(numbers), max_len, dtype=torch.long)
        padded_colors = torch.zeros(len(colors), max_len, dtype=torch.long)
        for i, (n, c) in enumerate(zip(numbers, colors)):
            padded_numbers[i, : len(n)] = n
            padded_colors[i, : len(c)] = c
        return padded_numbers, padded_colors, torch.stack(targets)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = SchurActorCritic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train and save
    model = train_transformer(model, dataloader, epochs=100)

    torch.save(model.state_dict(), "schur_transformer.pt")

    # Verify with loaded model
    model = load_model(model, "schur_transformer.pt")

    test_seq = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 1),
        (7, 2),
        (8, 3),
        (9, 4),
        (10, 5),
    ]
    z = 11
    verify_transformer(model, test_seq, z)
