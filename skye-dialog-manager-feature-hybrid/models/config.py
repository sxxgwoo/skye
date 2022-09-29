
from easydict import EasyDict as edict
import torch
from pathlib import Path

cur = '/home/joon/HT.V0.3.27'

args = edict(
    {
        "model": "gpt2",
        "dataset_path": f"{cur}/data/situationchat_original.json",
        "dataset_cache": f"/home/joon/HT.V0.3.27/situationchat_original_dataset_cache_GPT2Tokenizer",
        "model_checkpoint": f"{cur}/checkpoint/DialoGPT-large/",
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 0.9,
        "max_history": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "no_sample": True,
        "max_length": 20,
        "min_length": 1,
        "seed": 0,
    }
)
