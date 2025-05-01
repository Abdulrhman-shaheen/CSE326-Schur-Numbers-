import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# Bitmask operations
def set_bit(mask, pos):
    return mask | (1 << pos)

def check_bit(mask, pos):
    return (mask & (1 << pos)) != 0

def is_valid_coloring(mask, z):
    for x in range(1, z // 2 + 1):
        y = z - x
        if check_bit(mask, x) and check_bit(mask, y):
            return False
    return True

# Neural network model
class SchurNet(nn.Module):
    def __init__(self):
        super(SchurNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * 64, 256)
        self.fc_policy = nn.Linear(256, 4)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc_policy(x), dim=-1)
        value = torch.tanh(self.fc_value(x))
        return policy, value

# MCTS Node
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

# MCTS Implementation
class MCTS:
    def __init__(self, model, device, c_puct=1.0):
        self.model = model
        self.device = device
        self.c_puct = c_puct

    def get_state_input(self, state):
        masks = np.zeros((1, 4, 64), dtype=np.float32)
        for i, mask in enumerate(state):
            for j in range(64):
                masks[0, i, j] = 1.0 if check_bit(mask, j) else 0.0
        return torch.tensor(masks, dtype=torch.float32).to(self.device)

    def select(self, node, z):
        while node.children:
            max_uct = -float('inf')
            best_child = None
            for child in node.children:
                if child.visits == 0:
                    uct = float('inf')
                else:
                    uct = (child.value / child.visits) + self.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                if uct > max_uct:
                    max_uct = uct
                    best_child = child
            node = best_child
            if not node.children:
                break
        return node

    def expand(self, node, z):
        state_input = self.get_state_input(node.state)
        with torch.no_grad():
            policy, _ = self.model(state_input)
        policy = policy.cpu().numpy()[0]

        for color in range(4):
            new_state = list(node.state)
            new_state[color] = set_bit(new_state[color], z)
            if is_valid_coloring(new_state[color], z):
                child = MCTSNode(tuple(new_state), parent=node, action=color)
                child.prior = policy[color]
                node.children.append(child)
        if not node.children:
            return False
        return True

    def simulate(self, node, z, max_z=44):
        if z >= max_z:
            return 1.0
        state = list(node.state)
        current_z = z + 1
        while current_z <= max_z:
            valid_colors = []
            for color in range(4):
                new_mask = set_bit(state[color], current_z)
                if is_valid_coloring(new_mask, current_z):
                    valid_colors.append(color)
            if not valid_colors:
                return -1.0
            color = random.choice(valid_colors)
            state[color] = set_bit(state[color], current_z)
            current_z += 1
        return 1.0

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self, root_state, z, num_simulations=1000):
        root = MCTSNode(root_state)
        for _ in range(num_simulations):
            node = self.select(root, z)
            if node.visits == 0 and self.expand(node, z):
                state_input = self.get_state_input(node.state)
                with torch.no_grad():
                    _, value = self.model(state_input)
                value = value.item()
            else:
                value = self.simulate(node, z)
            self.backpropagate(node, value)

        # Select best action
        best_action = None
        best_visits = -1
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = child.action
        return best_action

# Inference
def find_coloring(max_z=44):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchurNet().to(device)
    model.load_state_dict(torch.load('schur_model_weights.pth'))
    model.eval()
    mcts = MCTS(model, device)

    initial_state = (
        set_bit(set_bit(0, 1), 4),
        set_bit(set_bit(0, 2), 3),
        set_bit(0, 5),
        0
    )

    state = initial_state
    z = 5
    color_names = ['Red', 'Blue', 'Green', 'Cyan']

    print("Searching for coloring...")
    while z < max_z:
        action = mcts.search(state, z + 1, num_simulations=1000)
        if action is None:
            print(f"Failed to find valid coloring at z={z + 1}")
            break
        
        new_state = list(state)
        new_state[action] = set_bit(new_state[action], z + 1)
        state = tuple(new_state)
        print(f"z={z + 1}: Assigned to {color_names[action]}")
        z += 1

    if z == max_z:
        print("\nFinal Coloring:")
        for i, color in enumerate(color_names):
            numbers = [j for j in range(1, max_z + 1) if check_bit(state[i], j)]
            print(f"{color}: {numbers}")

if __name__ == "__main__":
    find_coloring(max_z=44)