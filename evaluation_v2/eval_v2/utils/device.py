import torch

def pick_device(requested: str = "auto") -> str:
    """
    requested: auto | cpu | cuda | mps | cuda:0 | mps | cpu
    Returns a string representing the device.
    """
    requested = (requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        # mps only on Apple Silicon + proper torch build
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # allow "cuda:0" etc.
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        return requested

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is False.")
        return "mps"

    if requested == "cpu":
        return "cpu"

    raise ValueError(f"Unknown device: {requested}")