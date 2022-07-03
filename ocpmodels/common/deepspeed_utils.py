import json


def initialize_deepspeed_data(*args, deepspeed_config=None):
    """
    Deepspeed initialization function for a neural network. Casts the input tensors to the data type
    that is needed by DeepSpeed for the configuration defined in self.deepspeed_config.
    This function should be specifically used for data tensors or quantities based on the data
    which are of floating point type and will not be changed during the forward pass.
    """
    if len(args) > 1:
        return (deepspeed_convert_type(x, deepspeed_config) for x in args)
    else:
        return deepspeed_convert_type(args, deepspeed_config)


def deepspeed_convert_type(x, deepspeed_config=None):
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
