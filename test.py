def insertion_sort(A):
    for j in range(len(A) - 1, -1, -1):
        key = A[j]
        i = j + 1
        while i < len(A) and A[i] < key:
            A[i - 1] = A[i]
            i = i + 1
            print(f"value of i: {i}")
        A[i - 1] = key
        print(f"Array at j = {j}: {A}")

# Example usage
my_list = [17, 8, 4, 25, 2, 10, 16, 22, 9, 15]
insertion_sort(my_list)
print(my_list)
