import time

import kp
import numpy as np


class MatMulOp:
    def __init__(self, manager: kp.Manager, tile_size: int = -1):
        self.mgr = manager

        props = self.mgr.get_device_properties()
        max_workgroup_invocation = props['max_work_group_invocations']
        max_workgroup_size = props['max_work_group_size']
        if tile_size < 0:
            tile_size = 1
            while (4 * tile_size * tile_size <= max_workgroup_invocation
                   and 2 * tile_size <= max_workgroup_size[0]
                   and 2 * tile_size <= max_workgroup_size[1]):
                tile_size *= 2

        assert tile_size > 0
        assert tile_size * tile_size <= max_workgroup_invocation
        assert tile_size <= max_workgroup_size[0]
        assert tile_size <= max_workgroup_size[1]
        self.tile_size = tile_size

        self.shader = '''
#version 450

layout (local_size_x = {tile_size}, local_size_y = {tile_size}) in;

layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};

layout (constant_id = 0) const float tensor_size_f = 0;

shared float sub_tensor_1[{tile_size}][{tile_size}];
shared float sub_tensor_2[{tile_size}][{tile_size}];

void main()
{{
    uint row = gl_LocalInvocationID.x; // 0 .. tile_size
    uint col = gl_LocalInvocationID.y; // 0 .. tile_size
    // gl_WorkGroupID : 0 .. tensor_size / tile_size
    uint globalRow = {tile_size} * gl_WorkGroupID.x + row;
    uint globalCol = {tile_size} * gl_WorkGroupID.y + col;

    uint tensor_size = uint(tensor_size_f);
    float acc = 0.0;
    uint numTiles = tensor_size / {tile_size};
    for(uint t = 0u; t < numTiles; t++)
    {{
        uint tiledRow = ({tile_size} * t) + row;
        uint tiledCol = ({tile_size} * t) + col;
        sub_tensor_1[col][row] = in_tensor_1[(tiledCol * tensor_size) + globalRow];
        sub_tensor_2[col][row] = in_tensor_2[(globalCol * tensor_size) + tiledRow];

        memoryBarrierShared();
        barrier();

        for(uint k = 0u; k < {tile_size}; k++)
            acc += sub_tensor_1[k][row] * sub_tensor_2[col][k];

        barrier();
    }}
    out_tensor[tensor_size * globalCol + globalRow] = acc;
}}'''
        self.compiled_shader = kp.Shader.compile_source(self.shader.format(tile_size=tile_size))
        self.tensor_shape: tuple[int, int] = (0, 0)
        self.params: list[kp.Tensor] = []
        self.algo = None

    def __call__(self, tensor_shape: tuple[int, int], tensor_in_1: kp.Tensor, tensor_in_2: kp.Tensor,
                 tensor_out: kp.Tensor):
        params = [tensor_in_1, tensor_in_2, tensor_out]

        if self.algo is None or self.tensor_shape != tensor_shape or self.params != params:
            self.tensor_shape = tensor_shape
            self.params = params
            tile_size = min(tensor_shape[0], tensor_shape[1], self.tile_size)
            self.compiled_shader = kp.Shader.compile_source(self.shader.format(tile_size=tile_size))
            workgroup = [tensor_shape[0] // tile_size, tensor_shape[1] // tile_size, 1]
            self.algo = self.mgr.algorithm(
                params,  # params
                self.compiled_shader,  # spirv
                workgroup,  # workgroup
                [float(tensor_shape[0])],  # spec_consts
                [])  # push_consts

        (self.mgr.sequence()
         .record(kp.OpTensorSyncDevice([tensor_in_1, tensor_in_2]))
         .record(kp.OpAlgoDispatch(self.algo))
         .record(kp.OpTensorSyncLocal([tensor_out]))
         .eval())


def main():
    mgr = kp.Manager()

    matmul_op = MatMulOp(mgr)

    tensor_size = 4096
    tensor_shape = [tensor_size, tensor_size]
    tensor_in_1 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_in_2 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_out = mgr.tensor(np.zeros(tensor_shape))

    print(f'{tensor_shape} input tensors:\n'
          f'{tensor_in_1.data().reshape(tensor_shape)}\n'
          f'{tensor_in_2.data().reshape(tensor_shape)}\n')

    matmul_op(tensor_shape, tensor_in_1, tensor_in_2, tensor_out)

    experiment_count = 8
    start_time = time.time()
    for _ in range(experiment_count):
        matmul_op(tensor_shape, tensor_in_1, tensor_in_2, tensor_out)
    end_time = time.time()
    experiment_time = end_time - start_time
    op_count = tensor_shape[0] * tensor_shape[1] * ((tensor_shape[1] * 2) - 1)

    print(f'Output :\n{tensor_out.data().reshape(tensor_shape)}')

    print(f'{experiment_count} matmul time : '
          f'{experiment_time * 1000:0.2f}ms => '
          f'{experiment_count / experiment_time:0.2f}op/s or '
          f'{experiment_count * op_count / (1e9 * experiment_time):0.2f} GFLOPS')


if __name__ == '__main__':
    main()
