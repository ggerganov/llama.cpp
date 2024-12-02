import kp
import numpy as np


def main():
    mgr = kp.Manager()

    tensor_size = 4
    tensor_shape = [tensor_size, tensor_size]
    tensor_in_1 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_in_2 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_out = mgr.tensor(np.zeros(tensor_shape))

    print(f'Input tensors:\n'
          f'{tensor_in_1.data().reshape(tensor_shape)}\n'
          f'{tensor_in_2.data().reshape(tensor_shape)}\n')

    params = [tensor_in_1, tensor_in_2, tensor_out]

    matmul_shader = kp.Shader.compile_source('''
#version 450

layout (local_size_x = 1, local_size_y = 1) in;

layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float in_tensor_2[]; };
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float tensor_size_f = 0;


void main()
{
    uint globalRow = gl_GlobalInvocationID.x;
    uint globalCol = gl_GlobalInvocationID.y;
    uint tensor_size = uint(tensor_size_f);
    float acc = 0.0;
    for(uint k = 0u; k < tensor_size; k++)
        acc += in_tensor_1[(k * tensor_size) + globalRow] * in_tensor_2[(globalCol * tensor_size) + k];
    out_tensor[(globalCol * tensor_size) + globalRow] = acc;
}''')

    algo = mgr.algorithm(
        params,  # params
        matmul_shader,  # spirv
        (*tensor_shape, 1),  # workgroup
        [float(tensor_size)],  # spec_consts
        [])  # push_consts

    (mgr.sequence()
     .record(kp.OpTensorSyncDevice(params))
     .record(kp.OpAlgoDispatch(algo))
     .record(kp.OpTensorSyncLocal(params))
     .eval())

    print(f'Output :\n{tensor_out.data().reshape(tensor_shape)}')


if __name__ == '__main__':
    main()
