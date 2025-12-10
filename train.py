import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")

import os, sys
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from utils.model_utils import setup_seed
from utils.data_utils import load_config, save_config
from trainers.finally_trainer import FinallyTrainerStageOne, FinallyTrainer

if __name__ == "__main__":
    # Load base config
    config = load_config()

    # Merge with CLI arguments
    conf_cli = OmegaConf.from_cli()
    config = OmegaConf.merge(config, conf_cli)

    # Save config to run directory
    exp_dir = getattr(config.exp, "exp_dir", ".")  # default to current dir
    run_dir = os.path.join(exp_dir, "checkpoints", config.exp.run_name)

    # Create the directory
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, run_dir)

    # Set seed
    setup_seed(config.exp.seed)

    # Initialize trainer
    if config.exp.stage == 1:
        trainer = FinallyTrainerStageOne(config)
    else:
        trainer = FinallyTrainer(config)

    # Setup and start training
    trainer.setup_training()
    trainer.training_loop()
