import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.nn.utils.weight_norm` is deprecated"
)

import os, sys
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from utils.model_utils import setup_seed
from utils.data_utils import load_config
from training.trainers.hifigan_trainer import gan_trainers_registry

if __name__ == "__main__":
    # Load base config
    config = load_config()

    # Merge with CLI arguments
    conf_cli = OmegaConf.from_cli()
    config = OmegaConf.merge(config, conf_cli)

    # Save config to run directory
    exp_dir = getattr(config.exp, "exp_dir", ".")
    run_dir = os.path.join(exp_dir, "inference_out", config.exp.run_name)
    print(f"Saving inference outputs to: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(run_dir, "config.yaml"))

    # Set seed
    setup_seed(config.exp.seed)

    # Initialize and run inference
    trainer = gan_trainers_registry[config.inference.trainer](config)
    trainer.setup_inference()
    trainer.inference()
