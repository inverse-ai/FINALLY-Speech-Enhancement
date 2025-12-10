import inspect
from trainers import losses

class LossBuilder:
    def __init__(self, device, losses_config):
        self.device = device
        self.losses_config = losses_config
        
        for loss_name, loss_config in losses_config.items():
            if loss_name == 'wavlm_loss':
                self.wavlm_loss = losses.WavLMLoss().to(device)
            elif loss_name == 'stft_loss':
                self.stft_loss = losses.STFTLoss().to(device)
            elif loss_name == 'gen_loss':
                self.gen_loss = losses.GeneratorLoss().to(device)
            elif loss_name == 'disc_loss':
                self.disc_loss = losses.DiscriminatorLoss().to(device)
            elif loss_name == 'feature_loss':
                self.feature_loss = losses.FeatureLoss().to(device)
            elif loss_name == 'pesq':
                self.pesq_loss = losses.PesqLoss_().to(device)
            elif loss_name == 'utmos':
                self.utmos_loss = losses.UTMOSLoss(device=device).to(device)
            else:
                raise ValueError(f'Unknown loss name: {loss_name}')

    def calculate_loss(self, info, tl_suffix=None):
        loss_dict = {}
        total_loss_str = 'total_loss'
        if tl_suffix is not None:
            total_loss_str += '_' + tl_suffix
        loss_dict[total_loss_str] = 0.0

        calculated_losses_set = set()

        for loss_name, loss_config in self.losses_config.items():
            coef = loss_config['coef']
            if loss_name == 'wavlm_loss':
                loss = self.wavlm_loss
            elif loss_name == 'stft_loss':
                loss = self.stft_loss
            elif loss_name == 'gen_loss':
                loss = self.gen_loss
            elif loss_name == 'disc_loss':
                loss = self.disc_loss
            elif loss_name == 'feature_loss':
                loss = self.feature_loss
            elif loss_name == 'l1_loss':
                loss = self.l1_loss
            elif loss_name == 'pesq':
                loss = self.pesq_loss
            elif loss_name == 'utmos':
                loss = self.utmos_loss
            else:
                raise ValueError(f'Unknown loss name: {loss_name}')
                
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
    
            loss_dict[total_loss_str] += coef * current_loss

        not_calculated_losses = list(set(self.losses_config.keys()).difference(calculated_losses_set))
        if len(not_calculated_losses) > 0:
            raise RuntimeWarning(f'Losses {not_calculated_losses} from config was not calculated.'+
                                'Possibly beacuse of was not given enough arguments for that.')

        return loss_dict[total_loss_str], loss_dict

