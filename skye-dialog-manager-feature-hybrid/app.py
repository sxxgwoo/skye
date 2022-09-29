# -*- coding: utf-8 -*-
""" O2O API """

# python
import sys

# 3rd-party
import hydra
from omegaconf import DictConfig

# framework
from utils import print_config
from sts import sts_init
import analysis_class
import openAI

def launch_server(config):
    server = hydra.utils.instantiate(config.server)
    app = server.app
    app.app['config'] = config
    sts_init(app.app)
    analysis_class.init()
    openAI.init()


    app.run(port=config.server.app_port)

@hydra.main(config_path="configs/", config_name="serve.yaml")
def hydra_entry(config: DictConfig):
    if config.work_dir not in sys.path:
        sys.path.append(config.work_dir)  # for vscode debug
    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)
    launch_server(config)

if __name__ == '__main__':
    hydra_entry()

