import datetime
import os
import sys
import torch
import typer
import multiprocess

from dpdl.cli import cli
from dpdl.logger_config import configure_logger

def detect_gpus(log):
    if torch.cuda.is_available():
        log.info(f"CUDA is available. Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    elif torch.has_mps:
        log.info("MPS (Apple Metal) is available, but this is specific to Apple Silicon GPUs.")
    else:
        log.info("No CUDA or MPS GPUs detected.")

    # Check for AMD GPUs via ROCm
    try:
        rocminfo_output = subprocess.check_output("rocminfo", shell=True).decode()
        log.info("ROCm is installed. Here is the output from 'rocminfo':")
        log.info(rocminfo_output)
    except Exception as e:
        log.info("ROCm not detected or not properly installed.")
        log.info(f"Error: {e}")

def main():
    log = configure_logger()

    world_size = os.getenv('WORLD_SIZE')
    rank = os.getenv('RANK')
    local_rank = os.getenv('LOCAL_RANK')

    if world_size is None or local_rank is None or rank is None:
        log.error(
            "Script not correctly started: Environment variables 'WORLD_SIZE', 'RANK', and 'LOCAL_RANK' missing."
        )
        sys.exit(1)

    world_size = int(world_size)
    local_rank = int(local_rank)
    rank = int(rank)

    log.info(
        f'Rank {rank} initializing - our world size is {world_size} and local rank is {local_rank}.'
    )


    backend_engine = 'nccl' if torch.cuda.is_available() else 'gloo'
    # Initialize the process group
    torch.distributed.init_process_group(
        backend=backend_engine,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(
            minutes=60
        ),  # Transformations can be slow, increase timeout
    )

    # print info about GPUs
    detect_gpus(log)
    log.info(f'Rank {rank} initialized, with {backend_engine} as backend.')

    if torch.distributed.get_rank() == 0:
        log.info('All ranks initialized.')

    typer.run(cli)

    torch.distributed.destroy_process_group()

    log.info('And we are done!')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('--help')

        typer.run(cli)

    torch.set_float32_matmul_precision('high')

    # Enable TensorFloat-32 for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Reproducible results
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    # Fix Huggingface datasets map to work with multiple proceses.
    torch.set_num_threads(1)

    # Set to spawn so HF datasets map work with distributed
    multiprocess.set_start_method('spawn', force=True)

    if '--help' in sys.argv or '-h' in sys.argv:
        typer.run(cli)

    main()
