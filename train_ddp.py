"""
Distributed Data Parallel (DDP) Training Script for FINALLY

Launch with:
    torchrun --nproc_per_node=2 train_ddp.py exp.config_path=configs/stage3_config.yaml exp.run_name=my_run

Or with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_ddp.py exp.config_path=configs/stage3_config.yaml
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")

import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from utils.model_utils import setup_seed
from utils.data_utils import load_config, save_config
from trainers.finally_trainer_ddp import FinallyTrainerStageOneDDP, FinallyTrainerDDP


def setup_distributed():
    """Initialize distributed training environment."""
    # Get distributed info from environment (set by torchrun)
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        # Initialize the process group with explicit device_id
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            device_id=device  # Explicitly specify device to avoid hangs
        )

        if rank == 0:
            print(f"Distributed training initialized: {world_size} GPUs")

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)

    # Load base config
    config = load_config()

    # Merge with CLI arguments
    conf_cli = OmegaConf.from_cli()
    config = OmegaConf.merge(config, conf_cli)

    # Override device with local_rank for DDP
    config.exp.device = f'cuda:{local_rank}'

    # Add distributed info to config
    config.distributed = OmegaConf.create({
        'enabled': world_size > 1,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_main_process': is_main_process
    })

    # Save config to run directory (only main process)
    if is_main_process:
        exp_dir = getattr(config.exp, "exp_dir", ".")
        run_dir = os.path.join(exp_dir, "checkpoints", config.exp.run_name)
        os.makedirs(run_dir, exist_ok=True)
        save_config(config, run_dir)

    # Set seed (different for each rank to ensure different data)
    setup_seed(config.exp.seed + rank)

    # Initialize trainer
    if config.exp.stage == 1:
        trainer = FinallyTrainerStageOneDDP(config)
    else:
        trainer = FinallyTrainerDDP(config)

    # Setup and start training
    trainer.setup_training()

    try:
        trainer.training_loop()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
