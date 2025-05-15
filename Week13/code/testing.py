# Loads the model and try to test it. A simple interfacec.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from SchurTransfomer import load_model, SchurTransformer, verify_transformer


model = SchurTransformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(model, "data/schur_transformer.pt")

test_seq = [(1, 1), (2, 2), (3, 2), (4, 1), (5, 3), (6, 3), (7, 3), (8, 5), (9, 5), (10, 5), (11, 5), (12, 4), (13, 4), (14, 1), (15, 2), (16, 2), (17, 1), (18, 3), (19, 1), (20, 3), (21, 2), (22, 1), (23, 5), (24, 1), (25, 2), (26, 2), (27, 1), (28, 4), (29, 1), (30, 4), (31, 4), (32, 1), (33, 3), (34, 2), (35, 1), (36, 4), (37, 1), (38, 2), (39, 2), (40, 1), (41, 3), (42, 1), (43, 2), (44, 2), (45, 1), (46, 4), (47, 1), (48, 2), (49, 3), (50, 3), (51, 4), (52, 5), (53, 4), (54, 4), (55, 1), (56, 5), (57, 2), (58, 5), (59, 5), (60, 3), (61, 2), (62, 3), (63, 3), (64, 3), (65, 1), (66, 2), (67, 2), (68, 1), (69, 4), (70, 4), (71, 2), (72, 5), (73, 4), (74, 5), (75, 2), (76, 3), (77, 5), (78, 4), (79, 3), (80, 4)]

z = 81
valid = True

while valid:
    c, valid = verify_transformer(model, test_seq, z)
    test_seq.append((z, c))
    z += 1
