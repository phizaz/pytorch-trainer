import os
import json
from dataclasses import dataclass
from typing import List

@dataclass
class Env:
    cuda: List[int] = None
    global_lock: int = None
    num_workers: int = None

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            env = json.load(f)

        new = Env()
        for k, v in env.items():
            new.__dict__[k] = v
        return new

def read_env_file():
    for file in ENVFILES:
        if os.path.exists(file):
            return Env.from_json(file)
    return Env()

ENVFILES = [
    'mlkitenv.json',
    os.path.expanduser('~/mlkitenv.json'),
]
ENV = read_env_file()