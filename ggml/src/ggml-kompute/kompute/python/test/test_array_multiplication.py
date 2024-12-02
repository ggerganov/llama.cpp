import pyshader as ps
import kp
import numpy as np


def test_array_multiplication():

    # 1. Create Kompute Manager (selects device 0 by default)
    mgr = kp.Manager()

    # 2. Create Kompute Tensors to hold data
    tensor_in_a = mgr.tensor(np.array([2, 2, 2]))
    tensor_in_b = mgr.tensor(np.array([1, 2, 3]))
    tensor_out = mgr.tensor(np.array([0, 0, 0]))

    params = [tensor_in_a, tensor_in_b, tensor_out]

    # 4. Define the multiplication shader code to run on the GPU
    @ps.python2shader
    def compute_mult(index=("input", "GlobalInvocationId", ps.ivec3),
                                data1=("buffer", 0, ps.Array(ps.f32)),
                                data2=("buffer", 1, ps.Array(ps.f32)),
                                data3=("buffer", 2, ps.Array(ps.f32))):
        i = index.x
        data3[i] = data1[i] * data2[i]

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(mgr.algorithm(params, compute_mult.to_spirv())))
        .record(kp.OpTensorSyncLocal([tensor_out]))
        .eval())

    assert tensor_out.data().tolist() == [2.0, 4.0, 6.0]
    assert np.all(tensor_out.data() == [2.0, 4.0, 6.0])
