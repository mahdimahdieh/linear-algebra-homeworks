import numpy as np

print("\n4.")
vector1 = np.array([10, 20, 30])
vector2 = np.array([4, 5, 6])
sum_vector = vector1 + vector2
print("Vector 1:", vector1)
print("Vector 2:", vector2)
print("Sum:", sum_vector)

print("\n5.")
vector3 = np.array([7, 80, 900])
scaler = 7 * vector3
print("Vector 3:", vector3)
print("7 x Vector 3:", scaler)

print("\n6.")
vector4 = np.array([1, 2, 3, 4])
vector5 = np.array([5, 6, 7, 8])
dot_product = vector4 @ vector5
print("Vector 4:", vector4)
print("Vector 5:", vector5)
print("Dot Product:", dot_product)

print("\n7.")
vector6 = np.array([1, 2, 3])
vector7 = np.array([4, 5, 6])
vector8 = np.array([7, 8, 9])

product_of_sum = np.dot(vector6, (vector7 + vector8))

# Calculate right side: aᵀb + aᵀc
sum_of_product = np.dot(vector6, vector7) + np.dot(vector6, vector8)

print("Vector 6:", vector6)
print("Vector 7:", vector7)
print("Vector 8:", vector8)
print("Product of Sum:", product_of_sum)
print("Sum of Product:", sum_of_product)