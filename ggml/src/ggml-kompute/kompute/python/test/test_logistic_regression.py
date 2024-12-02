import pyshader as ps
import numpy as np
import kp

def test_logistic_regression():

    @ps.python2shader
    def compute_shader(
            index   = ("input", "GlobalInvocationId", ps.ivec3),
            x_i     = ("buffer", 0, ps.Array(ps.f32)),
            x_j     = ("buffer", 1, ps.Array(ps.f32)),
            y       = ("buffer", 2, ps.Array(ps.f32)),
            w_in    = ("buffer", 3, ps.Array(ps.f32)),
            w_out_i = ("buffer", 4, ps.Array(ps.f32)),
            w_out_j = ("buffer", 5, ps.Array(ps.f32)),
            b_in    = ("buffer", 6, ps.Array(ps.f32)),
            b_out   = ("buffer", 7, ps.Array(ps.f32)),
            l_out   = ("buffer", 8, ps.Array(ps.f32)),
            M       = ("buffer", 9, ps.Array(ps.f32))):

        i = index.x

        m = M[0]

        w_curr = vec2(w_in[0], w_in[1])
        b_curr = b_in[0]

        x_curr = vec2(x_i[i], x_j[i])
        y_curr = y[i]

        z_dot = w_curr @ x_curr
        z = z_dot + b_curr
        y_hat = 1.0 / (1.0 + exp(-z))

        d_z = y_hat - y_curr
        d_w = (1.0 / m) * x_curr * d_z
        d_b = (1.0 / m) * d_z

        loss = -((y_curr * log(y_hat)) + ((1.0 + y_curr) * log(1.0 - y_hat)))

        w_out_i[i] = d_w.x
        w_out_j[i] = d_w.y
        b_out[i] = d_b
        l_out[i] = loss


    mgr = kp.Manager(0)

    # First we create input and ouput tensors for shader
    tensor_x_i = mgr.tensor(np.array([0.0, 1.0, 1.0, 1.0, 1.0]))
    tensor_x_j = mgr.tensor(np.array([0.0, 0.0, 0.0, 1.0, 1.0]))

    tensor_y = mgr.tensor(np.array([0.0, 0.0, 0.0, 1.0, 1.0]))

    tensor_w_in = mgr.tensor(np.array([0.001, 0.001]))
    tensor_w_out_i = mgr.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    tensor_w_out_j = mgr.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    tensor_b_in = mgr.tensor(np.array([0.0]))
    tensor_b_out = mgr.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    tensor_l_out = mgr.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    tensor_m = mgr.tensor(np.array([ tensor_y.size() ]))

    # We store them in an array for easier interaction
    params = [tensor_x_i, tensor_x_j, tensor_y, tensor_w_in, tensor_w_out_i,
        tensor_w_out_j, tensor_b_in, tensor_b_out, tensor_l_out, tensor_m]

    mgr.sequence().eval(kp.OpTensorSyncDevice(params))

    # Create a managed sequence
    sq = mgr.sequence()

    # Record operation to sync memory from local to GPU memory
    sq.record(kp.OpTensorSyncDevice([tensor_w_in, tensor_b_in]))

    # Record operation to execute GPU shader against all our parameters
    sq.record(kp.OpAlgoDispatch(mgr.algorithm(params, compute_shader.to_spirv())))

    # Record operation to sync memory from GPU to local memory
    sq.record(kp.OpTensorSyncLocal([tensor_w_out_i, tensor_w_out_j, tensor_b_out, tensor_l_out]))

    ITERATIONS = 100
    learning_rate = 0.1

    # Perform machine learning training and inference across all input X and Y
    for i_iter in range(ITERATIONS):

        # Execute an iteration of the algorithm
        sq.eval()

        # Calculate the parameters based on the respective derivatives calculated
        for j_iter in range(tensor_b_out.size()):
            tensor_w_in.data()[0] -= learning_rate * tensor_w_out_i.data()[j_iter]
            tensor_w_in.data()[1] -= learning_rate * tensor_w_out_j.data()[j_iter]
            tensor_b_in.data()[0] -= learning_rate * tensor_b_out.data()[j_iter]

    assert tensor_w_in.data()[0] < 0.01
    assert tensor_w_in.data()[0] > 0.0
    assert tensor_w_in.data()[1] > 1.5
    assert tensor_b_in.data()[0] < 0.7
