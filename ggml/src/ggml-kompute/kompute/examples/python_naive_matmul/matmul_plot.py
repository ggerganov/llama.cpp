from argparse import ArgumentParser

import cv2
import numpy as np


def plot_tensor(window_name: str, tensor: np.ndarray, coord_highlight: tuple[int, int] = None):
    font_size = 48
    image = np.zeros((tensor.shape[1] * font_size, tensor.shape[0] * font_size, 3), dtype=np.uint8)

    for y in range(tensor.shape[1]):
        for x in range(tensor.shape[0]):
            if coord_highlight and x == coord_highlight[1] and y == coord_highlight[0]:
                cv2.putText(
                    image, str(int(tensor[y, x])), (x * font_size, int((y + 0.8) * font_size)),
                    cv2.FONT_HERSHEY_TRIPLEX, 1., (127, 127, 255))
            else:
                cv2.putText(
                    image, str(int(tensor[y, x])), (x * font_size, int((y + 0.8) * font_size)),
                    cv2.FONT_HERSHEY_TRIPLEX, 1., (255, 255, 255))

    cv2.imshow(window_name, image)


def main():
    parser = ArgumentParser()
    parser.add_argument('tensor_size', type=int, help='Size of the square tensors')
    parser.add_argument('tile_size', type=int)
    parser.add_argument('local_size', type=int, nargs=2)
    parser.add_argument('workgroup', type=int, nargs=2)
    arguments = parser.parse_args()

    tensor_size: int = arguments.tensor_size
    tile_size: int = arguments.tile_size
    local_size: tuple[int, int, int] = tuple(arguments.local_size)
    workgroup: tuple[int, int, int] = tuple(arguments.workgroup)

    tensor_shape = (tensor_size, tensor_size)
    tensor_1 = np.triu(np.ones(tensor_shape))
    tensor_2 = np.triu(np.ones(tensor_shape))
    tensor_out = np.zeros(tensor_shape)
    tensor_test_1 = np.zeros(tensor_shape)
    tensor_test_2 = np.zeros(tensor_shape)
    tensor_test_3 = np.zeros(tensor_shape)
    tensor_test_4 = np.zeros(tensor_shape)
    tensor_test_5 = np.zeros(tensor_shape)

    plot_tensor('tensor_1', tensor_1)
    plot_tensor('tensor_2', tensor_2)
    plot_tensor('tensor_out', tensor_out)
    plot_tensor('tensor_test_1', tensor_test_1)
    plot_tensor('tensor_test_2', tensor_test_2)
    plot_tensor('tensor_test_3', tensor_test_3)
    plot_tensor('tensor_test_4', tensor_test_4)
    plot_tensor('tensor_test_5', tensor_test_5)
    cv2.waitKey(-1)

    print(f'{workgroup=} {local_size=}')
    for workgroup_x in range(workgroup[0]):
        for workgroup_y in range(workgroup[1]):
            for invocation_x in range(workgroup_x * local_size[0], (workgroup_x + 1) * local_size[0]):
                for invocation_y in range(workgroup_y * local_size[1], (workgroup_y + 1) * local_size[1]):
                    row = invocation_x
                    col = invocation_y
                    globalRow = (tile_size * workgroup_x) + row
                    globalCol = (tile_size * workgroup_y) + col
                    try:
                        tensor_out[row, col] = row
                        tensor_test_1[row, col] = col
                        tensor_test_2[row, col] = workgroup_x
                        tensor_test_3[row, col] = workgroup_y
                        tensor_test_4[row, col] = globalRow
                        tensor_test_5[row, col] = globalCol
                        plot_tensor('tensor_out', tensor_out, (row, col))
                        plot_tensor('tensor_test_1', tensor_test_1, (row, col))
                        plot_tensor('tensor_test_2', tensor_test_2, (row, col))
                        plot_tensor('tensor_test_3', tensor_test_3, (row, col))
                        plot_tensor('tensor_test_4', tensor_test_4, (row, col))
                        plot_tensor('tensor_test_5', tensor_test_5, (row, col))
                        cv2.waitKey(-1)
                    except IndexError as error:
                        print(f'{workgroup_x=} {workgroup_y=} {row=} {col=}')
                        raise error

    plot_tensor('tensor_1', tensor_1)
    plot_tensor('tensor_2', tensor_2)
    plot_tensor('tensor_out', tensor_out)
    plot_tensor('tensor_test_1', tensor_test_1)
    plot_tensor('tensor_test_2', tensor_test_2)
    plot_tensor('tensor_test_3', tensor_test_3)
    plot_tensor('tensor_test_4', tensor_test_4)
    plot_tensor('tensor_test_5', tensor_test_5)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
