import json

from torch import Tensor
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
    tensors in the object are not detected by DeepSpeed. Although training seems to work without converting the Batch objects,
    the countless warnings might be bothersome. The Batch objects are converted in such a way that their type is a subclass
    of dict which implements all necessary operations performed on the data during training.

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
                new_batch_list = recursive_batch_to_dict(batch_list)
                return func(self, new_batch_list)
            else:
                return func(self, batch_list)

    return inner


def recursive_batch_to_dict(batch_list):
    """
    Convert an arbitrarily nested list/tuple of torch_geometric.data.batch.Batch objects as BatchDict so that they
    can be used just like dicts as well.
    """
    if isinstance(batch_list, (list, tuple)):
        for i, batch_list_item in enumerate(batch_list):
            batch_list[i] = recursive_batch_to_dict(batch_list_item)
        return batch_list
    elif isinstance(batch_list, Batch):
        return BatchDict(batch_list)


class BatchDict(dict):
    """
    Dict for forward pass input tensors that implements all operations needed on the data for training.
    Specifically, accessing the individual items as attributes is possible as well as pushing the dict
    with all of its tensor data to a different device.
    """

    def __init__(self, data):
        """
        Initialize the dict. Can be initialized from a torch_geometric.data.batch.Batch object or from
        an existing dict.
        """
        if isinstance(data, dict):
            super().__init__(data)
        elif isinstance(data, Batch):
            super().__init__(data.to_dict())
        else:
            raise ValueError(
                f"BatchDict cannot be initialized from a {type(data)} object!"
            )

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def to(self, device, *args, **kwargs):
        """
        Return an equivalent instance of BatchDict whose tensors contained in it are stored on the specified device.
        """
        return BatchDict(
            {
                key: self._attribute_to_device(value, device, *args, **kwargs)
                for key, value in self.items()
            }
        )

    def _attribute_to_device(self, object, device, *args, **kwargs):
        return (
            object.to(device, *args, **kwargs)
            if type(object) is Tensor
            else object
        )
