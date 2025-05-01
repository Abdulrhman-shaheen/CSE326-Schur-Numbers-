import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import uuid

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
        self.state = state  # Tuple of (red, blue, green, cyan) bitmasks
        self.parent = parent
        self.action = action  # Color index (0-3)
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
        # Convert state to model input: (1, 4, 64)
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

    def search(self, root_state, z, num_simulations=100):
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

        # Return policy (visit counts)
        policy = np.zeros(4)
        total_visits = sum(child.visits for child in root.children)
        if total_visits > 0:
            for child in root.children:
                policy[child.action] = child.visits / total_visits
        else:
            state_input = self.get_state_input(root.state)
            with torch.no_grad():
                policy, _ = self.model(state_input)
            policy = policy.cpu().numpy()[0]
        return policy

# Training loop
def train_model(max_z=20, episodes=100, simulations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchurNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    mcts = MCTS(model, device)

    initial_state = (
        set_bit(set_bit(0, 1), 4),  # Red: {1, 4}
        set_bit(set_bit(0, 2), 3),  # Blue: {2, 3}
        set_bit(0, 5),              # Green: {5}
        0                           # Cyan: {}
    )

    states = []
    policies = []
    values = []

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        state = initial_state
        z = 5
        episode_states = []
        episode_policies = []
        terminal_value = 0

        while z < max_z:
            policy = mcts.search(state, z, num_simulations=simulations)
            valid_colors = []
            for color in range(4):
                new_state = list(state)
                new_state[color] = set_bit(new_state[color], z + 1)
                if is_valid_coloring(new_state[color], z + 1):
                    valid_colors.append(color)
            
            if not valid_colors:
                terminal_value = -1
                break
            
            # Mask policy with valid colors
            policy = policy * np.array([1 if i in valid_colors else 0 for i in range(4)])
            policy_sum = policy.sum()
            
            # If policy sums to zero, assign uniform probabilities over valid colors
            if policy_sum < 1e-10:
                policy = np.array([1.0 / len(valid_colors) if i in valid_colors else 0 for i in range(4)])
            else:
                policy /= policy_sum

            # Choose action based on policy
            action = np.random.choice(4, p=policy)

            # Update state
            new_state = list(state)
            new_state[action] = set_bit(new_state[action], z + 1)
            state = tuple(new_state)

            episode_states.append(state)
            episode_policies.append(policy)
            z += 1

        # Assign rewards
        for s, p in zip(episode_states, episode_policies):
            states.append(s)
            policies.append(p)
            values.append(terminal_value)

    # Prepare training data
    X = np.zeros((len(states), 4, 64), dtype=np.float32)
    y_policy = np.array(policies)
    y_value = np.array(values, dtype=np.float32)

    for i, state in enumerate(states):
        for c, mask in enumerate(state):
            for j in range(64):
                X[i, c, j] = 1.0 if check_bit(mask, j) else 0.0

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_policy_tensor = torch.tensor(y_policy, dtype=torch.float32).to(device)
    y_value_tensor = torch.tensor(y_value, dtype=torch.float32).to(device)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_policy_tensor, y_value_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train model
    model.train()
    for epoch in range(10):
        total_policy_loss = 0
        total_value_loss = 0
        for batch_X, batch_policy, batch_value in dataloader:
            optimizer.zero_grad()
            policy_pred, value_pred = model(batch_X)
            policy_loss = policy_criterion(policy_pred, batch_policy)
            value_loss = value_criterion(value_pred.squeeze(), batch_value)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        print(f"Epoch {epoch + 1}/10, Policy Loss: {total_policy_loss/len(dataloader):.4f}, Value Loss: {total_value_loss/len(dataloader):.4f}")

    # Save model weights
    torch.save(model.state_dict(), 'schur_model_weights.pth')
    print("Model weights saved to schur_model_weights.pth")

if __name__ == "__main__":
    train_model(max_z=30, episodes=100, simulations=500)