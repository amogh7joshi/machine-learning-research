import copy
from typing import Union, List
from collections import OrderedDict

import numpy as np

import torch.nn as nn


def try_match_signature_to_torch_module(state_dict: OrderedDict) -> \
        Union[nn.Module, List, None]:
    """Matches a set of parameters to their corresponding torch layer."""
    params = state_dict.keys()

    # Characteristics of a batch normalization layer.
    if 'running_mean' in params or 'running_var' in params:
        return make_batchnorm(state_dict)

    # Check for Linear vs. Conv2d layers.
    if 'weight' in params:
        if len(state_dict['weight'].size()) == 2:
            return make_dense(state_dict)
        elif len(state_dict['weight'].size()) == 4:
            return make_conv(state_dict)

    # An unrecognized type of layer.
    return None

def make_conv(state_dict: OrderedDict) -> nn.Conv2d:
    out_planes, in_planes, kernel = state_dict['weight'].size()[0:3]
    use_bias = False
    if 'bias' in state_dict.keys():
        use_bias = True
    layer = nn.Conv2d(int(in_planes), int(out_planes),
                      kernel_size = (kernel, ) * 2, bias = use_bias)
    layer.load_state_dict(state_dict, strict = False)
    return layer

def make_dense(state_dict: OrderedDict) -> nn.Linear:
    out_planes, in_planes = state_dict['weight'].size()[:2]
    use_bias = False
    if 'bias' in state_dict.keys():
        use_bias = True
    layer = nn.Linear(int(in_planes), int(out_planes), bias = use_bias)
    layer.load_state_dict(state_dict)
    return layer

def make_batchnorm(state_dict: OrderedDict) -> nn.BatchNorm2d:
    layer = nn.BatchNorm2d(int(state_dict['weight'].size()[0]))
    layer.load_state_dict(state_dict, strict = False)
    return layer

def make_sequential(state_dict: OrderedDict) -> nn.Sequential:
    """Constructs an `nn.Sequential` module from the state dict."""
    layers = []
    state_keys = state_dict.keys()
    for layer_name in get_unique_layers(state_dict.keys()):
        corresponding_keys = get_corresponding_keys(layer_name, state_keys)
        layer_dict = construct_layer_dict(corresponding_keys, state_dict)
        maybe_layer = try_match_signature_to_torch_module(layer_dict)
        if maybe_layer is not None:
            if isinstance(maybe_layer, list):
                layers.extend(maybe_layer)
            layers.append(maybe_layer)
        else:
            layers.append(reconstruct_model(layer_dict))
    return nn.Sequential(*layers)

def copy_layer(module: nn.Module) -> nn.Module:
    """Copies an `nn.Module` whilst maintaining its parameters."""
    return copy.deepcopy(module)


def pop_prefix(string):
    """Returns a string with the prefix removed."""
    return '.'.join(string.split('.')[1:])

def has_sub_modules(key):
    """Returns whether a layer has any submodules."""
    return len(key.split('.')) != 1

def is_sequential(state_keys):
    """Returns whether the state dict is of a sequential model."""
    return all([i.split('.')[0].isdigit() for i in state_keys])

def is_prefixed_sequential(state_keys):
    """Returns whether the state dict is of a prefixed sequential model."""
    try:
        return all([i.split('.')[1].isdigit() for i in state_keys])
    except IndexError:
        return False

def get_unique_layers(state_keys):
    """Returns a list of unique layers given a set of state keys."""
    keys = [i.split('.')[0] for i in state_keys]
    indexes = np.unique(keys, return_index = True)[1]
    return [keys[idx] for idx in sorted(indexes)]

def get_sub_layers(state_keys):
    """Returns a list of all of the sub-layers in a set of state keys."""
    return [pop_prefix(key) for key in state_keys]

def get_corresponding_keys(key, state_keys):
    """Returns all keys in `state_keys` starting with `key`."""
    return [i for i in state_keys if i.split('.')[0] == key]

def construct_layer_dict(keys, state_dict) -> OrderedDict:
    """Construct a state dict for the layer (with the prefix removed)."""
    return OrderedDict([
        (pop_prefix(i[0]), i[1]) for i in state_dict.items() if i[0] in keys])


def reconstruct_model(state_dict):
    """Reconstructs a PyTorch `nn.Module` given a state dict.

    This method attempts to reconstruct the topology of a PyTorch model given
    only its state dict of weights, not the actual model itself. The returned
    model maintains the topology as provided in the state dict, but it is
    unable to account for any layers which do not have weights; for instance,
    activation or dropout layers do not appear in the new module.

    Parameters
    ----------
    state_dict : OrderedDict
        The state dict of a PyTorch model, in the format as returned by, for
        example, `<arbitrary model>.state_dict()`. This can also be a regular
        `dict`, but in order to maintain an accurate topology of the model it
        should be an `OrderedDict`.
    """
    # Get the keys of the state dict and all of the layers.
    state_keys = state_dict.keys()
    layers = get_unique_layers(state_keys)

    # Begin by checking whether we have received a sequential model.
    if is_sequential(state_dict):
        return make_sequential(state_dict)

    # Recursively construct each layer.
    module_layers = OrderedDict()
    for layer_name in layers:
        corresponding_keys = get_corresponding_keys(layer_name, state_keys)
        layer_dict = construct_layer_dict(corresponding_keys, state_dict)
        sub_layers = get_sub_layers(corresponding_keys)
        layer = None

        # If the module is sequential, then we need to repeat this procedure
        # for each of those layers and wrap the entire thing in a sequential model.
        if is_sequential(sub_layers):
            layer = make_sequential(layer_dict)

        # A custom case where there is a prefix followed by a sequential layer
        # naming format, e.g., `block.1...`, `block.2....`, etc.
        elif is_prefixed_sequential(sub_layers):
            prefix = sub_layers[0].split('.')[0]
            sub_dict = OrderedDict((pop_prefix(i[0]), i[1]) for i in layer_dict.items())
            seq_layer = make_sequential(sub_dict)
            layer = nn.Module()
            layer.add_module(prefix, seq_layer)

        # If `bn` is in the layer name, assume that it is a batch normalization
        # layer and use `nn.BatchNorm()` as the stand-in for the layer, unless
        # a custom layer is passed in the state dict.
        elif 'bn' in layer_name:
            if not is_sequential(sub_layers):
                layer = make_batchnorm(layer_dict)

        # If `conv` or `fc` is in the layer name, assume that it is either a `Conv2d`
        # layer or a `Linear` layers and check based on the actual model weights which
        # of the two it is, unless a user-passed argument is present.
        elif 'conv' in layer_name or 'fc' in layer_name:
            if not is_sequential(sub_layers):
                layer = try_match_signature_to_torch_module(layer_dict)

        # Otherwise, this is likely a custom module and we need to recursively
        # construct the module.
        else:
            mod = try_match_signature_to_torch_module(layer_dict)

            # If the result of `mod` is not None, then that means that
            # this method has almost certainly received the state dict
            # of an individual layer, so we can bypass the full `nn.Module`
            # construction here and just return the layer here.
            if mod is not None:
                return mod

            # Otherwise, keep attempting to construct recursive sub-layers
            # until we get to recognized layers. If no recognized layers are
            # found, then raise an error since that would be unsupported.
            sub_layer_dict = OrderedDict()
            for sub_layer in get_unique_layers(sub_layers):
                sub_layer_dict[sub_layer] = reconstruct_model(construct_layer_dict(
                    get_corresponding_keys(sub_layer, sub_layers), layer_dict))

            # Reconstruct a new `nn.Module` from the layer.
            layer = nn.Module()
            for name, mod in sub_layer_dict.items():
                layer.add_module(name, mod)

        # Update the layer dictionary.
        module_layers[layer_name] = layer

    # Reconstruct the `nn.Module` from the dictionary of layers.
    module = nn.Module()
    for name, mod in module_layers.items():
        module.add_module(name, mod)
    return module



