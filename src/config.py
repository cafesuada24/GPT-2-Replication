from omegaconf import ListConfig, OmegaConf, DictConfig


def load_config(config_path: str = r"configs/config.yaml") -> DictConfig | ListConfig:
    """Load and return config from a specified path

    Arguments:
    config_path -- yaml config file (default: 'configs/config.yaml')

    Return:
    DictConfig or ListConfig
    """

    return OmegaConf.load(config_path)


config = load_config()
