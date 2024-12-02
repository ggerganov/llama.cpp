import os

import kp
import numpy as np
import logging
import pyshader as ps

from .utils import compile_source

DIRNAME = os.path.dirname(os.path.abspath(__file__))

kp_log = logging.getLogger("kp")


def test_end_to_end():

    mgr = kp.Manager()

    tensor_in_a = mgr.tensor([2, 2, 2])
    tensor_in_b = mgr.tensor([1, 2, 3])
    # Explicit type constructor supports int, in32, double, float and int
    tensor_out_a = mgr.tensor_t(np.array([0, 0, 0], dtype=np.uint32))
    tensor_out_b = mgr.tensor_t(np.array([0, 0, 0], dtype=np.uint32))

    params = [tensor_in_a, tensor_in_b, tensor_out_a, tensor_out_b]

    shader = """
        #version 450

        layout (local_size_x = 1) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout(set = 0, binding = 1) buffer buf_in_b { float in_b[]; };
        layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };
        layout(set = 0, binding = 3) buffer buf_out_b { uint out_b[]; };

        // Kompute supports push constants updated on dispatch
        layout(push_constant) uniform PushConstants {
            float val;
        } push_const;

        // Kompute also supports spec constants on initalization
        layout(constant_id = 0) const float const_one = 0;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] += uint( in_a[index] * in_b[index] );
            out_b[index] += uint( const_one * push_const.val );
        }
    """

    workgroup = (3, 1, 1)
    spec_consts = [2]
    push_consts_a = [2]
    push_consts_b = [3]

    algo = mgr.algorithm(params, compile_source(shader), workgroup, spec_consts, push_consts_a)

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(algo))
        .record(kp.OpAlgoDispatch(algo, push_consts_b))
        .eval())

    sq = mgr.sequence()
    sq.eval_async(kp.OpTensorSyncLocal(params))

    sq.eval_await()

    assert tensor_out_a.data().tolist() == [4, 8, 12]
    assert tensor_out_b.data().tolist() == [10, 10, 10]


def test_shader_str():
    """
    Test basic OpAlgoBase operation
    """

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

    mgr = kp.Manager()

    tensor_in_a = mgr.tensor([2, 2, 2])
    tensor_in_b = mgr.tensor([1, 2, 3])
    tensor_out = mgr.tensor([0, 0, 0])

    params = [tensor_in_a, tensor_in_b, tensor_out]

    algo = mgr.algorithm(params, spirv)

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(algo))
        .record(kp.OpTensorSyncLocal(params))
        .eval())

    assert tensor_out.data().tolist() == [2.0, 4.0, 6.0]


def test_sequence():
    """
    Test basic OpAlgoBase operation
    """

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

    mgr = kp.Manager(0)

    tensor_in_a = mgr.tensor([2, 2, 2])
    tensor_in_b = mgr.tensor([1, 2, 3])
    tensor_out = mgr.tensor([0, 0, 0])

    params = [tensor_in_a, tensor_in_b, tensor_out]

    algo = mgr.algorithm(params, spirv)

    sq = mgr.sequence()

    sq.record(kp.OpTensorSyncDevice(params))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.record(kp.OpTensorSyncLocal(params))

    sq.eval()

    assert sq.is_init() == True

    sq.destroy()

    assert sq.is_init() == False

    assert tensor_out.data().tolist() == [2.0, 4.0, 6.0]
    assert np.all(tensor_out.data() == [2.0, 4.0, 6.0])

    tensor_in_a.destroy()
    tensor_in_b.destroy()
    tensor_out.destroy()

    assert tensor_in_a.is_init() == False
    assert tensor_in_b.is_init() == False
    assert tensor_out.is_init() == False


def test_pushconsts():

    spirv = compile_source("""
          #version 450
          layout(push_constant) uniform PushConstants {
            float x;
            float y;
            float z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { float pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          }
    """)

    mgr = kp.Manager()

    tensor = mgr.tensor([0, 0, 0])

    algo = mgr.algorithm([tensor], spirv, (1, 1, 1), [], [0.1, 0.2, 0.3])

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice([tensor]))
        .record(kp.OpAlgoDispatch(algo))
        .record(kp.OpAlgoDispatch(algo, [0.3, 0.2, 0.1]))
        .record(kp.OpAlgoDispatch(algo, [0.3, 0.2, 0.1]))
        .record(kp.OpTensorSyncLocal([tensor]))
        .eval())

    assert np.allclose(tensor.data(), np.array([0.7, 0.6, 0.5], dtype=np.float32))


def test_pushconsts_int():

    spirv = compile_source("""
          #version 450
          layout(push_constant) uniform PushConstants {
            int x;
            int  y;
            int  z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { int  pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          }
    """)

    mgr = kp.Manager()

    tensor = mgr.tensor_t(np.array([0, 0, 0], dtype=np.int32))

    spec_consts = np.array([], dtype=np.int32)
    push_consts = np.array([-1, -1, -1], dtype=np.int32)

    algo = mgr.algorithm([tensor], spirv, (1, 1, 1), spec_consts, push_consts)

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice([tensor]))
        .record(kp.OpAlgoDispatch(algo))
        .record(kp.OpAlgoDispatch(algo, np.array([-1, -1, -1], dtype=np.int32)))
        .record(kp.OpAlgoDispatch(algo, np.array([-1, -1, -1], dtype=np.int32)))
        .record(kp.OpTensorSyncLocal([tensor]))
        .eval())

    assert np.all(tensor.data() == np.array([-3, -3, -3], dtype=np.int32))


def test_workgroup():
    mgr = kp.Manager(0)

    tensor_a = mgr.tensor(np.zeros([16,8]))
    tensor_b = mgr.tensor(np.zeros([16,8]))

    @ps.python2shader
    def compute_shader_wg(gl_idx=("input", "GlobalInvocationId", ps.ivec3),
                          gl_wg_id=("input", "WorkgroupId", ps.ivec3),
                          gl_wg_num=("input", "NumWorkgroups", ps.ivec3),
                          data1=("buffer", 0, ps.Array(ps.f32)),
                          data2=("buffer", 1, ps.Array(ps.f32))):
        i = gl_wg_id.x * gl_wg_num.y + gl_wg_id.y
        data1[i] = f32(gl_idx.x)
        data2[i] = f32(gl_idx.y)

    algo = mgr.algorithm([tensor_a, tensor_b], compute_shader_wg.to_spirv(), (16,8,1))

    (mgr.sequence()
        .record(kp.OpTensorSyncDevice([tensor_a, tensor_b]))
        .record(kp.OpAlgoDispatch(algo))
        .record(kp.OpTensorSyncLocal([tensor_a, tensor_b]))
        .eval())

    print(tensor_a.data())
    print(tensor_b.data())

    assert np.all(tensor_a.data() == np.stack([np.arange(16)]*8, axis=1).ravel())
    assert np.all(tensor_b.data() == np.stack([np.arange(8)]*16, axis=0).ravel())


def test_mgr_utils():
    mgr = kp.Manager()

    props = mgr.get_device_properties()

    assert "device_name" in props

    devices = mgr.list_devices()

    assert len(devices) > 0
    assert "device_name" in devices[0]
