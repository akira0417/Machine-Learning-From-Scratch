import numpy as np


def max_pool2d(feature_map, pool_size=2, stride=2):
    height, width = feature_map.shape

    output_height = (
        height - pool_size
    ) // stride + 1

    output_width = (
        width - pool_size
    ) // stride + 1

    output = np.zeros(
        (output_height, output_width)
    )

    for i in range(output_height):
        for j in range(output_width):

            row = i * stride
            col = j * stride

            region = feature_map[
                row:row + pool_size,
                col:col + pool_size
            ]

            output[i, j] = np.max(region)

    return output