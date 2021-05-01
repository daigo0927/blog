import os
import time
from typing import Tuple, List
from omegaconf import OmegaConf, II
from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (64, 64)
    normalize: bool = True


@dataclass
class ModelConfig:
    image_size: Tuple[int, int] = II('preprocess.target_size')
    drop_rate: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class ExperimentConfig:
    preprocess: PreprocessConfig = PreprocessConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    logdir: str = 'outputs'


def preprocess(config: PreprocessConfig) -> None:
    print(OmegaConf.to_yaml(config))
    return


def build_model(config: ModelConfig) -> None:
    print(OmegaConf.to_yaml(config))
    return


def train(config: TrainConfig) -> None:
    print(OmegaConf.to_yaml(config))
    return


def run(config):
    preprocess(config.preprocess)
    build_model(config.model)
    train(config.train)
    return


def main() -> None:
    base_config = OmegaConf.structured(ExperimentConfig)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)

    # Create log directory
    time_str = time.strftime('%Y-%m-%dT%H-%M-%S')
    logdir = f'{config.logdir}/{time_str}'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Save additional and overall parameters
    OmegaConf.save(config, f'{logdir}/config.yaml')
    OmegaConf.save(cli_config, f'{logdir}/override.yaml')

    run(config)


if __name__ == '__main__':
    main()
