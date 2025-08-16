import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import collections
import inspect
import omegaconf
import dataclasses
import typing

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()


def move_to_device(x, device):
    if isinstance(x, (list, tuple)):
        x = x.__class__(move_to_device(t, device) for t in x)
    else:
        x = x.to(device)
    return x


def accuracy(real_logits=None, fake_logits=None):
    if real_logits is None and fake_logits is None:
        raise ValueError("at least one of the logits should be not None")

    real_acc = (
        real_logits.ge(0).float().mean() if real_logits is not None else 0.0
    )
    fake_acc = (
        fake_logits.le(0).float().mean() if fake_logits is not None else 0.0
    )
    if real_logits is None or fake_logits is None:
        return real_acc + fake_acc
    return (real_acc + fake_acc) / 2

def o(name, var):
    print(f"{name} : {var}")

class ClassRegistry:
    def __init__(self):
        self.classes = dict()
        self.args = dict()
        self.arg_keys = None

    def __getitem__(self, item):
        return self.classes[item]

    def make_dataclass_from_init(self, func, name, arg_keys):
        args = inspect.signature(func).parameters
        args = [
            (k, typing.Any, omegaconf.MISSING)
            if v.default is inspect.Parameter.empty
            else (k, typing.Optional[typing.Any], None)
            if v.default is None
            else (
                k,
                type(v.default),
                dataclasses.field(default=v.default),
            )
            for k, v in args.items()
        ]
        args = [
            arg
            for arg in args
            if (arg[0] != "self" and arg[0] != "args" and arg[0] != "kwargs")
        ]
        if arg_keys:
            self.arg_keys = arg_keys
            arg_classes = dict()
            for key in arg_keys:
                arg_classes[key] = dataclasses.make_dataclass(key, args)
            return dataclasses.make_dataclass(
                name,
                [
                    (k, v, dataclasses.field(default=v()))
                    for k, v in arg_classes.items()
                ],
            )
        return dataclasses.make_dataclass(name, args)

    def make_dataclass_from_classes(self, name):
        return dataclasses.make_dataclass(
            name,
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.classes.items()
            ],
        )

    def make_dataclass_from_args(self, name):
        return dataclasses.make_dataclass(
            name,
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.args.items()
            ],
        )

    def add_to_registry(self, name, arg_keys=None):
        def add_class_by_name(cls):
            self.classes[name] = cls
            self.args[name] = self.make_dataclass_from_init(
                cls.__init__, name, arg_keys
            )
            return cls

        return add_class_by_name
