from typing import List, Dict, Any
import numpy as np
import torch


def frontend_collate_fn(batch: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Collate B sequences, each a list of num_frames view dicts."""
    num_frames = len(batch[0])
    collated = []
    for frame_idx in range(num_frames):
        frame_dicts = [sample[frame_idx] for sample in batch]
        collated_frame = {}
        for key in frame_dicts[0]:
            vals = [fd[key] for fd in frame_dicts]
            if isinstance(vals[0], (torch.Tensor, np.ndarray)):
                collated_frame[key] = torch.stack([torch.as_tensor(v) for v in vals])
            elif isinstance(vals[0], str):
                collated_frame[key] = vals
            else:
                collated_frame[key] = vals
        collated.append(collated_frame)
    return collated
