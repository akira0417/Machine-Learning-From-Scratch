import numpy as np


def convolve2d(image, kernel, padding=0, stride=1):
    if padding > 0:
        image = np.pad(
            image,
            pad_width=padding,
            mode="constant",
            constant_values=0
        )

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1

    output = np.zeros(
        (output_height, output_width)
    )

    for i in range(output_height):
        for j in range(output_width):
            row = i * stride
            col = j * stride

            region = image[
                row:row + kernel_height,
                col:col + kernel_width
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output