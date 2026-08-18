import numpy as np

from convolution import convolve2d


image = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])


kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


output1 = convolve2d(
    image,
    kernel,
    padding=0,
    stride=1
)


output2 = convolve2d(
    image,
    kernel,
    padding=0,
    stride=2
)


print("Input:")
print(image.shape)

print("\nStride = 1:")
print(output1)
print("Shape:", output1.shape)

print("\nStride = 2:")
print(output2)
print("Shape:", output2.shape)