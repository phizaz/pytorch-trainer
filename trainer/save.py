import torch
import shutil
import uuid
import os

__all__ = ['safe_torch_save']

def safe_torch_save(obj, file_path, pickle_protocol=4):
    """safe torch save. It first writes into a tmp file then move it in later. 
    this prevents broken saves."""
    tmp_path = f'{file_path}.tmp.{uuid.uuid4()}'
    try:
        # protocol 4 is much faster to save large lists
        torch.save(obj, tmp_path, pickle_protocol=pickle_protocol)
        # remove the target path and replace with the new file
        shutil.move(tmp_path, file_path)
    except Exception:
        # remove the tmp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
