"""
Implements the AWQ for llama.cpp use cases.
Original paper: https://arxiv.org/abs/2306.00978

This code is based on versions of the AWQ implementation found in the following repositories:
* https://github.com/mit-han-lab/llm-awq
* https://github.com/casper-hansen/AutoAWQ
"""

import os
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.activations import GELUActivation


class ScaledActivation(nn.Module):
    """
    ScaledActivation module wraps an existing activation function and applies a
    scale factor to its output.

    Args:
        module (nn.Module): The activation function to be scaled.
        scales (torch.Tensor): A tensor of size (num_features,) containing the initial
            scale factors for each feature.

    Returns:
        torch.Tensor: The scaled output of the activation function.
    """

    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


def set_op_by_name(layer, name, new_module):
    """
    Set the new module for given module's name.

    Args:
        layer (nn.Module): The layer in which to replace the submodule.
        name (str): The path to the submodule to be replaced, using dot notation
            to access nested modules.
        new_module (nn.Module): The new module to replace the existing one.
    """
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_op_by_name(module, op_name):
    """
    Retrieves a submodule within a given layer based on its name.

    Args:
        module (nn.Module): The layer containing the submodule to find.
        op_name (str): The name of the submodule.

    Returns:
        nn.Module: The requested submodule found within the given layer.

    Raises:
        ValueError: If the specified submodule cannot be found within the layer.
    """
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """
    Scales the weights of a LayerNorm and a list of fully-connected layers proportionally.

    Args:
        ln (nn.LayerNorm): The LayerNorm module to be scaled.
        fcs (List[nn.Linear]): A list of fully-connected layers to be scaled.
        scales (torch.Tensor): A 1D tensor of size (num_features,).
    """

    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """
    Scales the weights of two fully-connected layers in a specific pattern.

    Args:
        fc1 (nn.Linear): The first fully-connected layer to be scaled.
        fc2 (nn.Linear): The second fully-connected layer to be scaled.
        scales (torch.Tensor): A 1D tensor of size (num_features,).
    """
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    """
    Scales the weight of a GELU activation and a fully-connected layer proportionally.

    Args:
        gelu (Union[nn.GELU, BloomGelu, GELUActivation]): The GELU activation module to be scaled.
        fc (nn.Linear): The fully-connected layer to be scaled.
        scales (torch.Tensor): A 1D tensor of size (num_features,).

    Raises:
        TypeError: If the `gelu` module is not of type `nn.GELU`, `BloomGelu`, or `GELUActivation`.
        TypeError: If the `fc` module is not of type `nn.Linear`.
    """
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


def apply_scale(module, scales_list, input_feat_dict=None):
    """
    Applies different scaling strategies to layers based on their type and hierarchy within a given module.

    Args:
        module (nn.Module): The module containing the layers to be scaled.
        scales_list (List[Tuple[str, List[str], torch.Tensor]]): A list of tuples containing:
            * prev_op_name (str): The name of the preceding operation or module,
                relative to which the layers to be scaled are located.
            * layer_names (List[str]): A list of names of the layers to be scaled, relative to the preceding operation.
            * scales (torch.Tensor): A 1D tensor of size (num_features,) containing the scaling factors for each feature.
        input_feat_dict (Optional[Dict[str, torch.Tensor]]): A dictionary mapping layer names to their corresponding
            input features (optional).
    """
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or "rmsnorm" in str(prev_op.__class__).lower():
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


@torch.no_grad()
def apply_clip(module, clip_list):
    """
    Applies element-wise clipping to the weight of a specific layer within a given module.

    Args:
        module (nn.Module): The module containing the layer to be clipped.
        clip_list (List[Tuple[str, torch.Tensor]]): A list of tuples containing:
            * name (str): The name of the layer to be clipped, relative to the root of the module.
            * max_val (torch.Tensor): A 1D or 2D tensor defining the upper bound for each element of the layer's weight.
    """
    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()


def add_scale_weights(model_path, scale_path, tmp_path):
    """
    Adds pre-computed Activation Weight Quantization (AWQ) results to a model,
    including scaling factors and clipping bounds.

    Args:
        model_path (str): Path to the pre-trained model to be equipped with AWQ.
        scale_path (str): Path to the AWQ scale factors (.pt file).
        tmp_path (str): Path to the temporary directory where the equipped model will be saved.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True
    )
    model.eval()
    awq_results = torch.load(str(scale_path), map_location="cpu")
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])
    model.save_pretrained(str(tmp_path))
    os.system(f"cp {str(model_path)}/tokenizer* {str(tmp_path)}")
