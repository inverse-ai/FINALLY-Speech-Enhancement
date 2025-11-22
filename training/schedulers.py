from tqdm import tqdm
from typing import Literal
from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

schedulers_registry = ClassRegistry()

WarmUpCurve = Literal['linear', 'convex', 'concave']
ReduceLrTime = Literal['epoch', 'step', 'period']

class WarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, step_period, warmup_steps, warmup_curve, base_scheduler, **kwargs):
        self.base_scheduler = base_scheduler
        self.step_period = step_period
        self.warmup_steps = warmup_steps
        self.finished_warmup = True
        self.current_step = 0

        def lr_lambda(current_step):
            if current_step <= warmup_steps:
                if warmup_curve == 'linear':
                    return current_step / max(1, warmup_steps)
                elif warmup_curve == 'convex':
                    return (current_step / max(1, warmup_steps))**4
                elif warmup_curve == 'concave':
                    return (current_step / max(1, warmup_steps))**0.25
            else:
                self.finished_warmup = True
                return 1.0

        super().__init__(optimizer, lr_lambda, **kwargs)

    def step(self, epoch=None):
        self.current_step += 1
        if self.finished_warmup:
            if self.current_step % self.step_period == 0:
                self.base_scheduler.step(epoch)
        else:
            super().step(epoch)

    def get_last_lr(self):
        if self.finished_warmup:
            return self.base_scheduler.get_last_lr()
        return super().get_last_lr()
    
class BaseScheduler:
    def __init__(self, optimizer, base_scheduler_type,
                reduce_time: ReduceLrTime, step_period=None, warmup_steps=0, warmup_curve='linear', **kwargs):
        base_scheduler = base_scheduler_type(optimizer, **kwargs)
        self.reduce_time = reduce_time
        if reduce_time == 'period':
            self.period = step_period

        self.scheduler = WarmUpScheduler(optimizer, step_period, warmup_steps, warmup_curve, base_scheduler)
        # if warmup_steps > 0:
        # else:
        #     self.scheduler = base_scheduler

    def step(self, epoch=None):
        self.scheduler.step(epoch)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

@schedulers_registry.add_to_registry(name='exponential')
class ExponentialScheduler(BaseScheduler):
    def __init__(
            self,
            optimizer,
            reduce_time: ReduceLrTime,
            step_period=None,
            warmup_steps=0,
            warmup_curve: WarmUpCurve='linear',
            **kwargs
        ):
        super().__init__(optimizer, ExponentialLR, reduce_time, step_period, warmup_steps, warmup_curve, **kwargs)
