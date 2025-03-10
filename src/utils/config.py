"""This module loads configurations"""

from omegaconf import ListConfig, OmegaConf, DictConfig


def load_config(config_path: str = r"configs/config.yaml") -> DictConfig | ListConfig:
    """Load and return config from a specified path

    Arguments:
    config_path -- yaml config file (default: 'configs/config.yaml')

    Return:
    DictConfig or ListConfig
    """

    conf = OmegaConf.load(config_path)

    conf.model.config.update(conf.model_configs[conf.model.model_size])

    return conf

config = load_config()
