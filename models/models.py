from torch import nn
from utils.class_registry import ClassRegistry

LRELU_SLOPE = 0.1
models_registry = ClassRegistry()

@models_registry.add_to_registry(name='identity')
class IdentityModel(nn.Module):
    def forward(self, wav, *args, **kwargs):
        return wav.clone()