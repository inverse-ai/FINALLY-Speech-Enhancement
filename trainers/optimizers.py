from torch.optim import AdamW

class AdamW_(AdamW):
    def __init__(self, params, lr=0.001, beta1=0.5, beta2=0.999):
        super().__init__(params, lr=lr, betas=(beta1, beta2))
