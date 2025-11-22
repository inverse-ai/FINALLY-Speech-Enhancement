from utils.class_registry import ClassRegistry
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW

optimizers_registry = ClassRegistry()

@optimizers_registry.add_to_registry(name='adam')
class Adam_(Adam):
    def __init__(self, params, lr=0.001, beta1=0.5, beta2=0.999):
        super().__init__(params, lr=lr, betas=(beta1, beta2))

@optimizers_registry.add_to_registry(name='sgd')
class SGD_(SGD):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init_(params, lr=lr, momentum=momentum)

@optimizers_registry.add_to_registry(name='adamW')
class AdamW_(AdamW):
    def __init__(self, params, lr=0.001, beta1=0.5, beta2=0.999):
        super().__init__(params, lr=lr, betas=(beta1, beta2))
