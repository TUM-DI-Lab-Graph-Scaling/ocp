import json


def deepspeed_forward(forward):
    """
    Decorates a forward function of a GNN that takes a DataBatch object as input.
    The wrapped function that is returned casts the torch.tensor attributes of the
    input DataBatch object to the data type that is needed by DeepSpeed for the configuration
    defined in self.deepspeed_config.
    """
    def new_forward(self, data):
        data.pos = deepspeed_type_converter(data.pos, self.deepspeed_config)
        return forward(self, data)
    return new_forward


def deepspeed_tensor_initialization(initializer):
    """
    Decorates a parameter initialization function for a neural network that takes arbitrary
    inputs and returns a torch.tensor. The wrapped function that is returned casts the output
    to the data type that is needed by DeepSpeed for the configuration defined in 
    self.deepspeed_config.
    """
    def new_initializer(self, *args, **kwargs):
        return deepspeed_type_converter(initializer(self, *args, **kwargs), self.deepspeed_config)
    return new_initializer


def deepspeed_type_converter(x, deepspeed_config=None):
    """
    Converts torch tensors to the needed data type for the specified deepspeed config 
    json file. If no file path is passed, just the tensor itself will be returned.
    """
    if deepspeed_config is None:
        return x
    else:
        with open(deepspeed_config, "r") as config_file:
            config = json.load(config_file)
            if "fp16" in config and config["fp16"]["enabled"]:
                return x.half()
            elif "bf16" in config and config["bf16"]["enabled"]:
                return x.bfloat16()
            else:
                return x



