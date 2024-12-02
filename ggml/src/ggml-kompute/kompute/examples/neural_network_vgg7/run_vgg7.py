import kp
import numpy
import os
import sys
import time
import sh_conv
import sh_common

if len(sys.argv) != 3:
    print("run_vgg7.py INPUT OUTPUT")
    print(" Tiling is not implemented, but padding is implemented")
    sys.exit(1)

# NOTES:
# + Tiling is not implemented, but padding is implemented
#   So don't run anything too big through it

if False:
    kpm = kp.Manager(1)
    if kpm.get_device_properties()["device_name"].count("RAVEN") > 0:
        raise "Safety cut-out triggered. Sorry!"
else:
    kpm = kp.Manager()

image = sh_common.image_load(sys.argv[1])
image = image.repeat(2, 0).repeat(2, 1)
image = numpy.pad(image, [[7, 7], [7, 7], [0, 0]], mode = "edge")

# Ensure image has 4 channels even though they will be unused.
# This is because of vectorization vec4 magic.
while image.shape[2] < sh_common.VSZ:
    image = numpy.pad(image, [[0, 0], [0, 0], [0, 1]], mode = "constant")

# sh_common.image_save("pad.png", image)

# Prepare the initial tensor.

tensor_in = kpm.tensor(image)
tensor_in_h = image.shape[0]
tensor_in_w = image.shape[1]
tensor_in_cg = 1
tensor_in_c = 3

# Run things.
channels = [32, 32, 64, 64, 128, 128, 3]

for i in range(7):
    # Prepare tensors.
    # 'c' is the total amount of channels, while 'cg' is the amount of vec4s (channel-groups).
    # This is important because weights have to be padded for the shader.
    tensor_out_h = tensor_in_h - 2
    tensor_out_w = tensor_in_w - 2
    tensor_out_c = channels[i]
    tensor_out_cg = (channels[i] + (sh_common.VSZ - 1)) // sh_common.VSZ
    # TODO: How to produce a blank tensor we don't care about the contents of?
    # This isn't being synced, and from experience so far that should handle most of it,
    #  but what about memory usage?
    # *Most* of these tensors live entirely on-device except when debugging.
    # Can that be handled? (Also good question: Does it even need to be handled?)
    tensor_out = kpm.tensor(numpy.zeros((tensor_out_h * tensor_out_w * tensor_out_cg * sh_common.VSZ)))
    weight = kpm.tensor(sh_common.load_weights_padded("kipper", (i * 2) + 0, tensor_out_c, tensor_in_c, 3))
    bias = kpm.tensor(sh_common.load_biases_padded("kipper", (i * 2) + 1, tensor_out_c))
    # Compute.
    # TODO: It'd be nice to wrap this up into a class for optimization purposes.
    workgroup = ((tensor_out_w + 7) // 8, (tensor_out_h + 1) // 2, tensor_out_cg)
    alg = kpm.algorithm(
        # tensors
        [tensor_in, bias, weight, tensor_out],
        # spirv
        sh_conv.conv_shader,
        # workgroup
        workgroup,
        # spec_consts
        [tensor_in_w, tensor_in_h, tensor_in_cg, tensor_out_w, tensor_out_h, tensor_out_cg],
        # push_consts
        []
    )

    print("Step complexity " + str(workgroup))
    print("Step channel layout " + str(tensor_in_cg) + " " + str(tensor_out_cg))

    # Do this first. Keep in mind "syncs" are copies.
    last_seq = kpm.sequence()
    things_to_sync_to_device = [bias, weight]
    if i == 0:
        # For first layer, the input isn't on-device yet
        things_to_sync_to_device.append(tensor_in)
    last_seq.eval_async(kp.OpTensorSyncDevice(things_to_sync_to_device))
    last_seq.eval_await()

    # Prepare
    seq = (kpm.sequence()
        .record(kp.OpAlgoDispatch(alg, []))
    )
    # Run
    seq.eval()

    print("Done with step")

    if False:
        # DEBUG:
        # We want to see the output, copy it to local
        last_seq = kpm.sequence()
        last_seq.eval_async(kp.OpTensorSyncLocal([tensor_out]))
        last_seq.eval_await()
        tensor_out.data().astype("<f4", "C").tofile("raw" + str(i) + ".bin")

    # Swap over.
    tensor_in = tensor_out
    tensor_in_h = tensor_out_h
    tensor_in_w = tensor_out_w
    tensor_in_c = tensor_out_c
    tensor_in_cg = tensor_out_cg

# Download output
fin_seq = kpm.sequence()
fin_seq.eval_async(kp.OpTensorSyncLocal([tensor_in]))
fin_seq.eval_await()

# Output
out_na = tensor_in.data().reshape((tensor_in_h, tensor_in_w, tensor_in_cg * sh_common.VSZ))
# Crop off 'alpha'
out_na = out_na[:, :, 0:3]
sh_common.image_save(sys.argv[2], out_na)

