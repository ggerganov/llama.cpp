from PIL import Image
import numpy

# just in case
global VSZ
VSZ = 4

def image_load(path) -> numpy.ndarray:
    """
    Loads an image.
    Doesn't Tensor it, in case you need to do further work with it.
    Shape is (h, w, 3).
    """
    # file
    na = numpy.array(Image.open(path))
    # change type
    na = na.astype("float32") / 255.0
    return na

def image_save(path, na: numpy.ndarray):
    """
    Saves an image.
    However, note this expects a numpy array.
    Shape is (h, w, 3).
    """
    # change type
    na = numpy.fmax(numpy.fmin(na * 255.0, 255), 0).astype("uint8")
    # file
    Image.fromarray(na).save(path)

def load_param(mdl, idx, expected):
    npa = numpy.fromfile("model-" + mdl + "/snoop_bin_" + str(idx) + ".bin", "<f4")
    assert npa.shape[0] == expected
    return npa

def save_param(mdl, idx, data):
    data.astype("<f4", "C").tofile("model-" + mdl + "/snoop_bin_" + str(idx) + ".bin")

def load_weights_padded(mdl, idx, tensor_out_c, tensor_in_c, weight_s):
    tensor_out_cg = (tensor_out_c + 3) // 4
    tensor_in_cg = (tensor_in_c + 3) // 4
    # weight & bias load & fixup
    # a reminder:
    #  [outputChannels][inputChannels][kernelH][kernelW]
    # ->
    #  [outputCGroups][kernelH][kernelW][inputCGroups][outputChannels][inputChannels]
    weight_na = load_param(mdl, idx, tensor_out_c * tensor_in_c * weight_s * weight_s)
    # start by putting in the initial shape
    weight_na = weight_na.reshape(tensor_out_c, tensor_in_c, weight_s, weight_s)
    # then by padding
    # NOTE: It is *critically important* that weight padding is done with the "zero" mode.
    # The shader WILL NOT ignore these values, but zeroing them causes them to have no effect.
    if (tensor_in_c & 3) != 0:
        weight_na = numpy.pad(weight_na, [[0, 0], [0, 4 - (tensor_in_c & 3)], [0, 0], [0, 0]], mode = "constant")
    if (tensor_out_c & 3) != 0:
        weight_na = numpy.pad(weight_na, [[0, 4 - (tensor_out_c & 3)], [0, 0], [0, 0], [0, 0]], mode = "constant")
    # reshape to finish splitting things up
    weight_na = weight_na.reshape(tensor_out_cg, 4, tensor_in_cg, 4, weight_s, weight_s)
    # result is:
    # [outputCGroups][outputChannels][inputCGroups][inputChannels][kernelH][kernelW]
    # and move output channels to the right...
    weight_na = numpy.moveaxis(weight_na, 1, 5)
    # result is:
    # [outputCGroups][inputCGroups][inputChannels][kernelH][kernelW][outputChannels]
    # and move input channels to the right...
    weight_na = numpy.moveaxis(weight_na, 2, 5)
    # result is:
    # [outputCGroups][inputCGroups][kernelH][kernelW][outputChannels][inputChannels]
    # and move input cgroups to the right...
    weight_na = numpy.moveaxis(weight_na, 1, 3)
    return weight_na

def load_biases_padded(mdl, idx, tensor_out_c):
    tensor_out_cg = (tensor_out_c + 3) // 4
    # [outputCGroups][outputChannels]
    # biases merely need padding
    # Again, has to be zero
    bias_na = load_param(mdl, idx, tensor_out_c)
    if (tensor_out_c & 3) != 0:
        bias_na = numpy.pad(bias_na, [[0, 4 - (tensor_out_c & 3)]], mode = "constant")
    return bias_na

