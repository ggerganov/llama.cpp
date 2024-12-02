import time

import kp
import numpy as np


class MatMulOp:
    def __init__(self, manager: kp.Manager, local_size_x: int = -1, local_size_y: int = -1):
        self.mgr = manager

        props = self.mgr.get_device_properties()
        max_workgroup_invocation = props['max_work_group_invocations']
        max_workgroup_size = props['max_work_group_size']
        if local_size_x < 1:
            if local_size_y > 0:
                local_size_x = 1
                while (2 * local_size_x * local_size_y <= max_workgroup_invocation
                       and 2 * local_size_x <= max_workgroup_size[0]):
                    local_size_x *= 2
            else:
                local_size_x = 1
                local_size_y = 1
                while 2 * local_size_x * local_size_y <= max_workgroup_invocation:
                    if 2 * local_size_x <= max_workgroup_size[0]:
                        local_size_x *= 2
                    if 2 * local_size_y <= max_workgroup_size[1]:
                        local_size_y *= 2
                    elif 2 * local_size_x > max_workgroup_size[0]:  # stop if neither x nor y can be double
                        break
        elif local_size_y < 0:
            local_size_y = 1
            while (2 * local_size_x * local_size_y <= max_workgroup_invocation
                   and 2 * local_size_x <= max_workgroup_size[0]):
                local_size_y *= 2

        assert local_size_x > 0
        assert local_size_y > 0
        assert local_size_x * local_size_y <= max_workgroup_invocation
        assert local_size_x <= max_workgroup_size[0]
        assert local_size_y <= max_workgroup_size[1]
        self.local_size_x = local_size_x
        self.local_size_y = local_size_y

        self.shader = '''
#version 450

layout (local_size_x = {local_size_x}, local_size_y = {local_size_y}) in;

layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};

layout (constant_id = 0) const float tensor_size_f = 0;


void main()
{{
    uint globalRow = gl_GlobalInvocationID.x;
    uint globalCol = gl_GlobalInvocationID.y;
    uint tensor_size = uint(tensor_size_f);
    float acc = 0.0;
    for(uint k = 0u; k < tensor_size; k++)
        acc += in_tensor_1[(k * tensor_size) + globalRow] * in_tensor_2[(globalCol * tensor_size) + k];
    out_tensor[(globalCol * tensor_size) + globalRow] = acc;
}}'''
        self.compiled_shader = kp.Shader.compile_source(self.shader.format(
            local_size_x=self.local_size_x, local_size_y=self.local_size_y))
        self.tensor_shape: tuple[int, int] = (0, 0)
        self.params: list[kp.Tensor] = []
        self.algo = None

    def __call__(self, tensor_shape: tuple[int, int], tensor_in_1: kp.Tensor, tensor_in_2: kp.Tensor,
                 tensor_out: kp.Tensor):
        params = [tensor_in_1, tensor_in_2, tensor_out]

        if self.algo is None or self.tensor_shape != tensor_shape or self.params != params:
            self.tensor_shape = tensor_shape
            self.params = params
            local_size_x = min(self.local_size_x, tensor_shape[0])
            local_size_y = min(self.local_size_y, tensor_shape[1])
            self.compiled_shader = kp.Shader.compile_source(self.shader.format(
                local_size_x=local_size_x, local_size_y=local_size_y))
            workgroup = (tensor_shape[0] // local_size_x, tensor_shape[1] // local_size_y, 1)
            print(f'{workgroup=} {self.local_size_x=} {self.local_size_y=}')
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

    tensor_size = 4064
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
