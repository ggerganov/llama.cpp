import time

import kp
import numpy as np
from imp1_naive import MatMulOp as MatMulOp1
from imp2_tiled import MatMulOp as MatMulOp2
from imp3_better_tiling import MatMulOp as MatMulOp3


def main():
    mgr = kp.Manager()
    for tensor_size, experiment_count in [(512, 1000), (4096, 5)]:
        tensor_shape = [tensor_size, tensor_size]
        tensor_shape = [tensor_size, tensor_size]
        mat_1 = np.triu(np.ones(tensor_shape))
        mat_2 = np.triu(np.ones(tensor_shape))

        tensor_in_1 = mgr.tensor(mat_1)
        tensor_in_2 = mgr.tensor(mat_2)
        tensor_out = mgr.tensor(np.zeros(tensor_shape))
        if tensor_size <= 512:
            mat_result = mat_1 @ mat_2
        else:
            MatMulOp1(mgr)(tensor_shape, tensor_in_1, tensor_in_2, tensor_out)
            mat_result = tensor_out.data().reshape(tensor_shape)  # CPU is too slow for big sizes

        print(f'{tensor_shape} input tensors:\n'
              f'{mat_1}\n'
              f'{mat_2}\n')
        print(f'Output :\n{mat_result}')

        for MatMulOp in [MatMulOp1, MatMulOp2, MatMulOp3]:
            tensor_out.data()[:] = 0
            mgr.sequence().record(kp.OpTensorSyncDevice([tensor_out]))
            matmul_op = MatMulOp(mgr)
            matmul_op(tensor_shape, tensor_in_1, tensor_in_2, tensor_out)

            start_time = time.time()
            for _ in range(experiment_count):
                matmul_op(tensor_shape, tensor_in_1, tensor_in_2, tensor_out)
            end_time = time.time()
            experiment_time = end_time - start_time
            op_count = tensor_shape[0] * tensor_shape[1] * ((tensor_shape[1] * 2) - 1)

            # print(tensor_out.data().reshape(tensor_shape))
            if (tensor_out.data().reshape(tensor_shape) == mat_result).all():
                print(f'From {MatMulOp.__module__} : {experiment_count} matmul time : '
                      f'{experiment_time * 1000:0.2f}ms => '
                      f'{experiment_count / experiment_time:0.2f}op/s or '
                      f'{experiment_count * op_count / (1e9 * experiment_time):0.2f} GFLOPS')
            else:
                print(f'Test failed => output tensor is wrong :\n{tensor_out.data().reshape(tensor_shape)}')


if __name__ == '__main__':
    main()
