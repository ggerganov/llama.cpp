import os
import pytest
import kp
import numpy as np

from .utils import compile_source

VK_ICD_FILENAMES = os.environ.get("VK_ICD_FILENAMES", "")

def test_type_float():

    shader = """
        #version 450
        layout(set = 0, binding = 0) buffer tensorLhs {float valuesLhs[];};
        layout(set = 0, binding = 1) buffer tensorRhs {float valuesRhs[];};
        layout(set = 0, binding = 2) buffer tensorOutput { float valuesOutput[];};
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main()
        {
            uint index = gl_GlobalInvocationID.x;
            valuesOutput[index] = valuesLhs[index] * valuesRhs[index];
        }
    """

    spirv = compile_source(shader)

    arr_in_a = np.array([123., 153., 231.], dtype=np.float32)
    arr_in_b = np.array([9482, 1208, 1238], dtype=np.float32)
    arr_out = np.array([0, 0, 0], dtype=np.float32)

    mgr = kp.Manager()

    tensor_in_a = mgr.tensor(arr_in_a)
    tensor_in_b = mgr.tensor(arr_in_b)
    tensor_out = mgr.tensor(arr_out)

    params = [tensor_in_a, tensor_in_b, tensor_out]

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(mgr.algorithm(params, spirv)))
        .record(kp.OpTensorSyncLocal([tensor_out]))
        .eval())

    assert np.all(tensor_out.data() == arr_in_a * arr_in_b)


def test_type_float_double_incorrect():

    shader = """
        #version 450
        layout(set = 0, binding = 0) buffer tensorLhs {float valuesLhs[];};
        layout(set = 0, binding = 1) buffer tensorRhs {float valuesRhs[];};
        layout(set = 0, binding = 2) buffer tensorOutput { float valuesOutput[];};
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main()
        {
            uint index = gl_GlobalInvocationID.x;
            valuesOutput[index] = valuesLhs[index] * valuesRhs[index];
        }
    """

    spirv = compile_source(shader)

    arr_in_a = np.array([123., 153., 231.], dtype=np.float32)
    arr_in_b = np.array([9482, 1208, 1238], dtype=np.uint32)
    arr_out = np.array([0, 0, 0], dtype=np.float32)

    mgr = kp.Manager()

    tensor_in_a = mgr.tensor_t(arr_in_a)
    tensor_in_b = mgr.tensor_t(arr_in_b)
    tensor_out = mgr.tensor_t(arr_out)

    params = [tensor_in_a, tensor_in_b, tensor_out]

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(mgr.algorithm(params, spirv)))
        .record(kp.OpTensorSyncLocal([tensor_out]))
        .eval())

    assert np.all(tensor_out.data() != arr_in_a * arr_in_b)

@pytest.mark.skipif("broadcom" in VK_ICD_FILENAMES,
                    reason="Broadcom doesn't support double")
@pytest.mark.skipif("swiftshader" in VK_ICD_FILENAMES,
                    reason="Swiftshader doesn't support double")
def test_type_double():

    shader = """
        #version 450
        layout(set = 0, binding = 0) buffer tensorLhs { double valuesLhs[]; };
        layout(set = 0, binding = 1) buffer tensorRhs { double valuesRhs[]; };
        layout(set = 0, binding = 2) buffer tensorOutput { double valuesOutput[]; };
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main()
        {
            uint index = gl_GlobalInvocationID.x;
            valuesOutput[index] = valuesLhs[index] * valuesRhs[index];
        }
    """

    spirv = compile_source(shader)

    arr_in_a = np.array([123., 153., 231.], dtype=np.float64)
    arr_in_b = np.array([9482, 1208, 1238], dtype=np.float64)
    arr_out = np.array([0, 0, 0], dtype=np.float64)

    mgr = kp.Manager()

    tensor_in_a = mgr.tensor_t(arr_in_a)
    tensor_in_b = mgr.tensor_t(arr_in_b)
    tensor_out = mgr.tensor_t(arr_out)

    params = [tensor_in_a, tensor_in_b, tensor_out]

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(mgr.algorithm(params, spirv)))
        .record(kp.OpTensorSyncLocal([tensor_out]))
        .eval())

    print(f"Dtype value {tensor_out.data().dtype}")

    assert np.all(tensor_out.data() == arr_in_a * arr_in_b)

def test_type_int():

    shader = """
        #version 450
        layout(set = 0, binding = 0) buffer tensorLhs { int valuesLhs[]; };
        layout(set = 0, binding = 1) buffer tensorRhs { int valuesRhs[]; };
        layout(set = 0, binding = 2) buffer tensorOutput { int valuesOutput[]; };
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main()
        {
            uint index = gl_GlobalInvocationID.x;
            valuesOutput[index] = valuesLhs[index] * valuesRhs[index];
        }
    """

    spirv = compile_source(shader)

    arr_in_a = np.array([123, 153, 231], dtype=np.int32)
    arr_in_b = np.array([9482, 1208, 1238], dtype=np.int32)
    arr_out = np.array([0, 0, 0], dtype=np.int32)

    mgr = kp.Manager()

    tensor_in_a = mgr.tensor_t(arr_in_a)
    tensor_in_b = mgr.tensor_t(arr_in_b)
    tensor_out = mgr.tensor_t(arr_out)

    params = [tensor_in_a, tensor_in_b, tensor_out]

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(mgr.algorithm(params, spirv)))
        .record(kp.OpTensorSyncLocal([tensor_out]))
        .eval())

    print(f"Dtype value {tensor_out.data().dtype}")

    assert np.all(tensor_out.data() == arr_in_a * arr_in_b)

def test_type_unsigned_int():

    shader = """
        #version 450
        layout(set = 0, binding = 0) buffer tensorLhs { uint valuesLhs[]; };
        layout(set = 0, binding = 1) buffer tensorRhs { uint valuesRhs[]; };
        layout(set = 0, binding = 2) buffer tensorOutput { uint valuesOutput[]; };
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main()
        {
            uint index = gl_GlobalInvocationID.x;
            valuesOutput[index] = valuesLhs[index] * valuesRhs[index];
        }
    """

    spirv = compile_source(shader)

    arr_in_a = np.array([123, 153, 231], dtype=np.uint32)
    arr_in_b = np.array([9482, 1208, 1238], dtype=np.uint32)
    arr_out = np.array([0, 0, 0], dtype=np.uint32)

    mgr = kp.Manager()

    tensor_in_a = mgr.tensor_t(arr_in_a)
    tensor_in_b = mgr.tensor_t(arr_in_b)
    tensor_out = mgr.tensor_t(arr_out)

    params = [tensor_in_a, tensor_in_b, tensor_out]

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(mgr.algorithm(params, spirv)))
        .record(kp.OpTensorSyncLocal([tensor_out]))
        .eval())

    print(f"Dtype value {tensor_out.data().dtype}")

    assert np.all(tensor_out.data() == arr_in_a * arr_in_b)

def test_tensor_numpy_ownership():

    arr_in = np.array([1, 2, 3])

    m = kp.Manager()

    t = m.tensor(arr_in)

    # This should increment refcount for tensor sharedptr
    td = t.data()

    assert td.base.is_init() == True
    assert np.all(td == arr_in)

    del t

    assert td.base.is_init() == True
    assert np.all(td == arr_in)

    m.destroy()

    assert td.base.is_init() == False
