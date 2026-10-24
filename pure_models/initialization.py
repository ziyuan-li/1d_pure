import math

import torch
from torch import nn


def make_linear(in_features, out_features, bias=True, dtype=torch.float32):
    layer = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    weights = (
        (layer.weight.real, layer.weight.imag)
        if layer.weight.is_complex()
        else (layer.weight,)
    )
    for weight in weights:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode="fan_in")
    if layer.bias is not None:
        bound = 1 / math.sqrt(in_features)
        biases = (
            (layer.bias.real, layer.bias.imag)
            if layer.bias.is_complex()
            else (layer.bias,)
        )
        for value in biases:
            nn.init.uniform_(value, -bound, bound)
    return layer
