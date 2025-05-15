import os


def aligned_array_print(*arrays):
    '''
    pretty print the arrays in a aligned way. like the one in the schur five paper.
    '''
    # Flatten and find the maximum number (N)
    N = max(num for arr in arrays for num in arr)

    # Create a set for each array for quick lookup
    sets = [set(arr) for arr in arrays]

    for s in sets:
        line = ""
        for i in range(1, N + 1):
            if i in s:
                line += f"{i:<3}"  # Print number left-aligned in 3-char space
            else:
                line += " " * 3    # Add spaces if number is missing
        print(line.rstrip())  # Strip trailing spaces for clean output


def check_bit(mask, pos):
    '''
        Becuase we're using 3 64 integers to represent the number.
    '''
    idx = pos // 64
    bit = pos % 64
    return (mask[idx] & (1 << bit)) != 0

def extract_coloring(s : str):
    '''
        return an array of 5 elemets, each element is a list of the numbers in that color. returns [[1,4],[2,3]] from 9 0 0, 6,0,0 (Binary representation: 1001, 0110)
     '''
    s = s.split(",")
    nums = [list(map(int, i.strip().split(" "))) for i in s]
    
    arrs = []

    for mask in nums:
        arr = []
        for i in range(128):
            if check_bit(mask, i):
                arr.append(i)
        arrs.append(arr)
    return arrs

def list_to_pairs(color_lists):
    '''
    takes the 2d array from extract_coloring and returns a list of tuples (number, color)
    '''
    pairs = []
    for color_idx, numbers in enumerate(color_lists, 1):
        for num in numbers:
            pairs.append((num, color_idx))
    return sorted(pairs)




def transform_data():
    '''
    Convert data from masks to pairs.
    '''
    data_dir = "/home/abdelrahman/schur-rl/data"

    for z in range(6, 81):
        print(f"Processing z = {z}")
        input_filename = f"states_z{z}.txt"
        output_filename = f"pairs_z{z}.txt"

        file_path = os.path.join(data_dir, input_filename)
        if not os.path.exists(file_path):
            continue

        output_file = os.path.join(data_dir, output_filename)
        
        with open(file_path, "r") as infile, open(output_file, "w") as outfile:
            lines = infile.readlines()  # Read all lines at once
            transformed_lines = []
            for line in lines:
                coloring = line[:-2]  # if this is actually required
                arrs = extract_coloring(coloring)
                pairs = list_to_pairs(arrs)
                transformed_lines.append(str(pairs))
            outfile.write("\n".join(transformed_lines) + "\n")


def valid(seq, z, c):
    '''
    Check if the current coloring is valid. for testings sometimes.
    '''
    for i in range(1, z // 2 + 1):
        if seq[i - 1][1] == c and seq[(z - i) - 1][1] == c :
            return False
    return True 



# Scratch Pad
seq = [(1, 1), (2, 2), (3, 2), (4, 1), (5, 3), (6, 3), (7, 3), (8, 5), (9, 5), (10, 5), (11, 5), (12, 4), (13, 4), (14, 1), (15, 2), (16, 2), (17, 1), (18, 3), (19, 1), (20, 3), (21, 2), (22, 1), (23, 5), (24, 1), (25, 2), (26, 2), (27, 1), (28, 4), (29, 1), (30, 4), (31, 4), (32, 1), (33, 3), (34, 2), (35, 1), (36, 4), (37, 1), (38, 2), (39, 2), (40, 1), (41, 3), (42, 1), (43, 2), (44, 2), (45, 1), (46, 4), (47, 1), (48, 2), (49, 3), (50, 4), (51, 4), (52, 4), (53, 4), (54, 5), (55, 5), (56, 5), (57, 5), (58, 5), (59, 5), (60, 5), (61, 5), (62, 3), (63, 3), (64, 3), (65, 1), (66, 2), (67, 2), (68, 4), (69, 4), (70, 4), (71, 2), (72, 3)]

print(valid(seq, 73, 2))

coloring = "181596204974098 2 0,308705172226060 140 0,13835623212850086112 257 0,16958939555966976 112 0,4593671619926298368 512 0"

arrs = extract_coloring(coloring)
pairs = list_to_pairs(arrs)
print(arrs)

aligned_array_print(*arrs)

# transform_data()

