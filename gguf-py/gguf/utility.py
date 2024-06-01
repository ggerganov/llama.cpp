from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from torch import Tensor


def fill_templated_filename(filename: str, encoding_scheme: str):
    # Given a file name fill in any type templates e.g. 'some-model-name.{ftype}.gguf'
    ftype_uppercase: str = encoding_scheme.upper()
    ftype_lowercase: str = encoding_scheme.lower()
    return filename.format(ftype_lowercase,
                           outtype=ftype_lowercase, ftype=ftype_lowercase,
                           OUTTYPE=ftype_uppercase, FTYPE=ftype_uppercase)


def per_model_weight_count_estimation(tensors: Iterator[tuple[str, Tensor]], expert_count: int) -> int:
    # TODO: Ensure parameter count is accurate throughout various model type
    #       May currently overestimate parameter count in Mamba model because
    #       output weights is tied with token embeddings.
    sum_weight_estimate = 0
    for name, data_torch in tensors:
        # Got A Tensor

        # We don't need these
        if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue

        # Calculate Tensor Volume
        sum_weights_in_tensor = 1
        for dim in data_torch.shape:
            sum_weights_in_tensor *= dim

        # Add Tensor Volume To Running Count
        sum_weight_estimate += sum_weights_in_tensor

    # Calculate weight estimate per model
    per_model_weight_estimate = (sum_weight_estimate / expert_count) if (expert_count > 0) else sum_weight_estimate

    return per_model_weight_estimate


def model_weight_count_rounded_notation(model_params_count: int) -> str:
    if model_params_count > 1e15 :
        # Quadrillion Of Parameters
        scaled_model_params = model_params_count * 1e-15
        scale_suffix = "Q"
    elif model_params_count > 1e12 :
        # Trillions Of Parameters
        scaled_model_params = model_params_count * 1e-12
        scale_suffix = "T"
    elif model_params_count > 1e9 :
        # Billions Of Parameters
        scaled_model_params = model_params_count * 1e-9
        scale_suffix = "B"
    elif model_params_count > 1e6 :
        # Millions Of Parameters
        scaled_model_params = model_params_count * 1e-6
        scale_suffix = "M"
    else:
        # Thousands Of Parameters
        scaled_model_params = model_params_count * 1e-3
        scale_suffix = "K"
    return f"{round(scaled_model_params)}{scale_suffix}"


def naming_convention(model_name: str, base_name: str, finetune_string:str, version_string:str, expert_count_int:int, model_params_count: int, encoding_scheme: str) -> str:
    # Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#gguf-naming-convention

    if base_name is not None:
        name = base_name.strip().title().replace(' ', '-')
    elif model_name is not None:
        name = model_name.strip().title().replace(' ', '-')
    else:
        name = "ggml-model"

    per_model_rounded_weight_estimate = model_weight_count_rounded_notation(model_params_count)
    if expert_count_int is not None and expert_count_int > 0:
        parameters = f"-{expert_count_int}x{per_model_rounded_weight_estimate}"
    else:
        parameters = f"-{per_model_rounded_weight_estimate}"

    finetune = f"-{finetune_string.strip().title().replace(' ', '-')}" if finetune_string is not None else ""

    version = f"-{version_string.strip().replace(' ', '-')}" if version_string is not None else ""

    encoding = f"-{encoding_scheme.strip().replace(' ', '-').upper()}"

    return f"{name}{parameters}{finetune}{version}{encoding}"
