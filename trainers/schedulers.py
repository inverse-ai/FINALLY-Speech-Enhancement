from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

class WarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, start_step, step_period, warmup_steps, base_scheduler, **kwargs):
        self.base_scheduler = base_scheduler
        self.step_period = step_period
        self.warmup_steps = warmup_steps
        self.current_step = start_step

        def lr_lambda(current_step):
            if current_step <= warmup_steps:
                return current_step / max(1, warmup_steps)
            else:
                return 1.0

        super().__init__(optimizer, lr_lambda, **kwargs)

    def step(self, epoch=None):
        self.current_step += 1
        if self.current_step > self.warmup_steps:
            if self.current_step % self.step_period == 0:
                self.base_scheduler.step(epoch)
        else:
            super().step(epoch)

    def get_last_lr(self):
        if self.current_step > self.warmup_steps:
            return self.base_scheduler.get_last_lr()
        return super().get_last_lr()
    
class BaseScheduler:
    def __init__(self, optimizer, start_step, base_scheduler_type, step_period=None, warmup_steps=0, **kwargs):
        base_scheduler = base_scheduler_type(optimizer, **kwargs)
        
        self.period = step_period

        self.scheduler = WarmUpScheduler(optimizer, start_step, step_period, warmup_steps, base_scheduler)

    def step(self, epoch=None):
        self.scheduler.step(epoch)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

class ExponentialScheduler(BaseScheduler):
    def __init__(
            self,
            optimizer,
            start_step,
            step_period=None,
            warmup_steps=0,
            **kwargs
        ):
        super().__init__(optimizer, start_step, ExponentialLR, step_period, warmup_steps, **kwargs)
