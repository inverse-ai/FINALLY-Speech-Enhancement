import torch
from training.trainers.base_trainer import BaseTrainer
from training.trainers.base_trainer import gan_trainers_registry

@gan_trainers_registry.add_to_registry(name='dummy_trainer')
class DummyTrainer(BaseTrainer):
    def synthesize_wavs(self, batch):        
        result_dict = {
            'gen_wav': [],
            'name': []
        }

        for name, input_wav in zip(batch['name'], batch['input_wav']):
            gen_wav = input_wav.to(self.device)
            result_dict['gen_wav'].append(gen_wav)
            result_dict['name'].append(name)
    
        result_dict['gen_wav'] = torch.stack(result_dict['gen_wav'])
        return result_dict
    