import json

from torch_geometric.data.batch import Batch


def initialize_deepspeed_data(*args, deepspeed_config=None):
    """
    Deepspeed initialization function for a neural network. Casts the input tensors to the data type
    that is needed by DeepSpeed for the configuration defined in self.deepspeed_config.
    This function should be specifically used for data tensors or quantities based on the data
    which are of floating point type and will not be changed during the forward pass.

    Args:
        *args (List[torch.Tensor]): Tensors to be typecasted.
        deepspeed_config (Path, optional): Path to the DeepSpeed config file in which
            the information about type conversions is stored.
    """
    if len(args) > 1:
        return (deepspeed_convert_type(x, deepspeed_config) for x in args)
    elif len(args) == 1:
        return deepspeed_convert_type(args[0], deepspeed_config)


def deepspeed_convert_type(x, deepspeed_config=None):
    """
    Converts torch tensors to the needed data type for the specified deepspeed config
    json file. If no file path is passed, just the tensor itself will be returned.

    Args:
        x (torch.Tensor): Tensor to be converted.
        deepspeed_config (Path, optional): Path to the DeepSpeed config file in which
            the information about type conversions is stored.
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


def deepspeed_trainer_forward(func):
    """
    Decorator for a forward function using inputs of type torch_geometric.data.batch.Batch. This object is not suited for
    DeepSpeed Stage 3 training and using it as input of a forward function will cause multiple warning printouts since the
    tensors in the object are not detected by DeepSpeed. Although training seems to work without wrapping the Batch objects,
    the countless warnings might be bothersome. The Batch objects are wrapped in such a way that they are of type dict as
    well. As a dict, the parameters can be detected by DeepSpeed.

    Args:
        func: Method of a class taking self and batch_list (aribitrarily nested lists/tuples of torch_geometric.data.batch.Batch
        objects).
    """

    def inner(self, batch_list):
        if (
            "deepspeed_config" not in self.config
            or self.config["deepspeed_config"] is None
        ):
            return func(self, batch_list)

        with open(self.config["deepspeed_config"], "r") as config_file:
            config = json.load(config_file)
            if (
                "zero_optimization" in config
                and config["zero_optimization"]["stage"] == 3
            ):
                # do batch_list conversion
                new_batch_list = recursive_batch_wrap(batch_list)
                return func(self, new_batch_list)
            else:
                return func(self, batch_list)

    return inner


def recursive_batch_wrap(batch_list):
    """
    Wrap an arbitrarily nested list/tuple of torch_geometric.data.batch.Batch objects as Stage3BatchWrapper so that they
    can be used just like dicts as well.
    """
    if isinstance(batch_list, (list, tuple)):
        for i, batch_list_item in enumerate(batch_list):
            batch_list[i] = recursive_batch_wrap(batch_list_item)
        return batch_list
    elif isinstance(batch_list, Batch):
        return Stage3BatchWrapper(batch_list)


class Stage3BatchWrapper(dict):
    """
    Wrapper for a torch_geometric.data.batch.Batch object enabling dict functionality.
    """

    def __init__(self, batch):
        super().__init__(batch.to_dict())
        self._batch = batch

    def __getattr__(self, attribute):
        return getattr(self._batch, attribute)

    def to(self, device):
        return Stage3BatchWrapper(self._batch.to(device))
