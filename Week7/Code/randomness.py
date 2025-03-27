import random


def assign_color(coloring, num):
    for color_array in coloring.values():
        if can_place(sorted(color_array), num):
            color_array.append(num)
            return coloring
    return None


def can_place(color_array, num):
    i, j = 0, len(color_array) - 1
    while i < j:
        if color_array[i] + color_array[j] == num:
            return False
        elif color_array[i] + color_array[j] < num:
            i += 1
        else:
            j -= 1
    return True


def add_number(coloring, num_to_add):
    # Deep copy
    coloring_copy = {color: color_array[:] for color, color_array in coloring.items()}

    output = assign_color(coloring_copy, num_to_add)
    if output:
        return output  # Success on first try

    # Flatten all elements with their current colors
    color_elements = [(color, elem) for color, elems in coloring_copy.items() for elem in elems]

    seen = set()
    num_tries = 0

    while output is None and len(seen) < len(color_elements):
        element_to_reassign = random.choice(color_elements)

        if element_to_reassign in seen:
            continue
        seen.add(element_to_reassign)

        orig_color, elem = element_to_reassign
        coloring_copy[orig_color].remove(elem)

        reassigned = False
        for color, color_array in coloring_copy.items():
            if color == orig_color:
                continue
            if can_place(color_array, elem):
                color_array.append(elem)
                temp = assign_color(coloring_copy, num_to_add)
                if temp:
                    output = temp
                    reassigned = True
                    break
                else:
                    # Revert if it didn't help
                    color_array.remove(elem)

        if not reassigned:
            coloring_copy[orig_color].append(elem)

        num_tries += 1

    print("Tries:", num_tries)
    return output


# Test setup
coloring = {
    "red": [1, 4, 7, 10, 12],
    "blue": [2, 3, 11],
    "purple": [5, 6, 8, 9]
}

num_to_add = 13

result = add_number(coloring, num_to_add)

if result:
    print("Successfully updated coloring:")
    for color, arr in result.items():
        print(f"{color}: {sorted(arr)}")
else:
    print("Failed to update coloring.")