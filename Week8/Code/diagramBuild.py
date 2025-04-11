from graphviz import Digraph

class Node:
    def __init__(self, color_assignments, added_number=None, added_color=None):
        self.color_assignments = color_assignments
        self.added_number = added_number
        self.added_color = added_color
        self.children = []

def is_valid_addition(color, n, color_assignments):
    numbers = color_assignments[color]
    for i in range(len(numbers)):
        for j in range(i, len(numbers)):
            if numbers[i] + numbers[j] == n:
                return False
    return True

def build_tree(node, current_number=6):
    if current_number > 13:  # Extended to 13
        return
    
    for color in ['red', 'blue', 'green']:
        if is_valid_addition(color, current_number, node.color_assignments):
            new_colors = {
                'red': node.color_assignments['red'].copy(),
                'blue': node.color_assignments['blue'].copy(),
                'green': node.color_assignments['green'].copy()
            }
            new_colors[color].append(current_number)
            child = Node(new_colors, current_number, color)
            node.children.append(child)
            build_tree(child, current_number + 1)

def create_highres_diagram(root):
    dot = Digraph(comment='Extended Recursion Diagram', format='png')
    dot.attr(rankdir='TB', nodesep='0.8', ranksep='1.2', dpi='300')  # Improved spacing and resolution
    node_counter = [0]
    
    def add_nodes(node, parent_id=None):
        current_id = str(node_counter[0])
        node_counter[0] += 1
        
        if parent_id is None:
            label = "Initial State:\n"
            label += "\n".join([f"{k}: {sorted(v)}" for k, v in node.color_assignments.items()])
        else:
            label = f"Add {node.added_number} to {node.added_color}\l"
            if node.children:
                label += "\nCurrent Colors:\l"
                for color in ['red', 'blue', 'green']:
                    label += f"{color}: {sorted(node.color_assignments[color])}\l"
            else:
                label += "\nFinal Coloring:\l"
                for color in ['red', 'blue', 'green']:
                    label += f"{color}: {sorted(node.color_assignments[color])}\l"
        
        # Node styling
        dot.node(current_id, label=label, shape='rectangle',
                fontname='Helvetica', fontsize='9',
                margin='0.3', width='2.5', height='0.8')  # Better text formatting
        
        if parent_id is not None:
            dot.edge(parent_id, current_id)
        
        for child in node.children:
            add_nodes(child, current_id)
    
    add_nodes(root)
    dot.render('highres_diagram', view=True, cleanup=True)

# Initialize and build extended tree
initial_state = {
    'red': [1, 4],
    'blue': [2, 3],
    'green': [5]
}
root = Node(initial_state)
build_tree(root)

# Generate improved visual diagram
print("Generating high-resolution diagram up to 13...")
create_highres_diagram(root)
print("PNG diagram saved as 'highres_diagram.png'")