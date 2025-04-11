import sys
import time 
sys.setrecursionlimit(10**6)


def validate(arr, num):
    i = 0
    j = len(arr) - 1

    while i <= j and j >= 0 and i < len(arr):
        if arr[i] + arr[j] == num:
            return 0
        elif arr[i] + arr[j] > num:
            j -= 1
        elif arr[i] + arr[j] < num:
            i += 1
    return 1


def recurse(red, blue, purple, green, indigo, num):
    if len(indigo) == 1:
        if indigo[0] == 45:
            print("Red:", red)
            print("Blue:", blue)
            print("Purple:", purple)
            print("Green:", green)
        return
            
    if validate(red, num):
        n_red = red + [num]
        recurse(n_red, blue, purple, green, indigo, num + 1)

    if validate(blue, num):
        n_blue = blue + [num]
        recurse(red, n_blue, purple, green, indigo, num + 1)

    if validate(purple, num):
        n_purple = purple + [num]
        recurse(red, blue, n_purple, green, indigo, num + 1)

    if validate(green, num):
        n_green = green + [num]
        recurse(red, blue, purple, n_green, indigo, num + 1)

    if validate(indigo, num):
        n_indigo = indigo + [num]
        recurse(red, blue, purple, green, n_indigo, num + 1)


def main():
    red = [1, 4]
    blue = [2, 3]
    purple = []
    green = []
    indigo = []
    

    num = 5
    t = time.time()
    recurse(red, blue, purple, green, indigo, num)
    print("Time taken:", time.time() - t)

main()
