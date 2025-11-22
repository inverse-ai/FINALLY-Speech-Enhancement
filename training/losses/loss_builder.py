import inspect
from utils.class_registry import ClassRegistry

losses_registry = ClassRegistry()

class LossBuilder:
    def __init__(self, device, losses_config):
        self.losses = {}
        self.coefs = {}
        self.device = device
        print("Initializing LossBuilder on device:", self.device)
        
        for loss_name, loss_config in losses_config.items():
            self.coefs[loss_name] = loss_config['coef']
            loss_args = loss_config['args'] if 'args' in loss_config else {}
            self.losses[loss_name] = losses_registry[loss_name](**loss_args, device=self.device).to(self.device)

    def calculate_loss(self, info, tl_suffix=None):
        loss_dict = {}
        total_loss_str = 'total_loss'
        if tl_suffix is not None:
            total_loss_str += '_' + tl_suffix
        loss_dict[total_loss_str] = 0.0

        calculated_losses_set = set()

        for loss_name, loss in self.losses.items():
            signature = inspect.signature(loss.forward)
            param_names = [param.name for param in signature.parameters.values()]

            current_loss = 0.0
            for suffix, kwargs in info.items():
                if suffix != '':
                    suffix = '_' + suffix

                loss_args = {param: kwargs[param] for param in param_names if param in kwargs}

                if len(loss_args) < len(param_names):
                    continue
                
                loss_val = loss(**loss_args)
                calculated_losses_set.add(loss_name)
                current_loss += loss_val
                loss_dict[loss_name + suffix] = float(loss_val)
    
            loss_dict[total_loss_str] += self.coefs[loss_name] * current_loss

        not_calculated_losses = list(set(self.losses.keys()).difference(calculated_losses_set))
        if len(not_calculated_losses) > 0:
            raise RuntimeWarning(f'Losses {not_calculated_losses} from config was not calculated.'+
                                'Possibly beacuse of was not given enough arguments for that.')

        return loss_dict[total_loss_str], loss_dict

