import os
import torch

def pick_device(requested: str = "auto") -> torch.device:
    """
    requested: auto | cpu | cuda | mps | cuda:0 | mps | cpu
    """
    requested = (requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # mps only on Apple Silicon + proper torch build
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # allow "cuda:0" etc.
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        return torch.device(requested)

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {requested}")