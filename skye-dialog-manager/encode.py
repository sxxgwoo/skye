# -*- coding: utf-8 -*-
""" O2O API """

# python
import os
import sys
import json

# 3rd-party
import hydra
import faiss
from omegaconf import DictConfig

# framework
from utils import print_config, get_logger

logger = get_logger(__name__)

def encode(config):
    logger.info("init encoder")
    encoder = hydra.utils.instantiate(config.model.encoder)
    for file in config.data.files:
        basename = os.path.basename(file)
        filepath = os.path.join(config.data.dir, file)
        index_file = basename + '.faiss'
        index_filepath = os.path.join(config.data.dir, index_file)
        logger.info(filepath)
        if(not config.data.overwrite):
            if os.path.exists(index_filepath):
                continue
        with open(filepath) as fp:
            db = json.load(fp)
            questions = [qas['question'] for qas in db['data']]
            encodings = encoder.encode(questions, show_progress_bar=True)      # db에서 question을 추출해 encoding작업
            index = faiss.IndexFlatIP(encodings.shape[-1])          # encoding한 question을 faiss 라이브러리를 이용해 추후 
            faiss.normalize_L2(encodings)            # embedding한 벡터를 정규화
            index.add(encodings)                     
            faiss.write_index(index, index_filepath)
            logger.info("INDEX filepath : {0}, {1} records".format(index_filepath, index.ntotal))


@hydra.main(config_path="configs/", config_name="encode.yaml")
def hydra_entry(config: DictConfig):
    if config.work_dir not in sys.path:
        sys.path.append(config.work_dir)  # for vscode debug
    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)
    encode(config)

if __name__ == '__main__':
    hydra_entry()
