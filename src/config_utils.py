"""Load the YAML config file into a normal Python dictionary."""

import yaml


def load_config(path):
    # Open the YAML file and turn it into a dictionary.
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
