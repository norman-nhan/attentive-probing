import torch
from transformers import AutoModel, AutoVideoProcessor

def count_parameters(model):
    """
    Examples:
    ```python
    print("model total parameters:", count_parameters(model))
    ```"""
    n_params = sum(p.numel() for p in model.parameters())

    if n_params >= 1e9:
        return f"{n_params / 1e9:.2f}B"
    elif n_params >= 1e6:
        return f"{n_params / 1e6:.2f}M"
    elif n_params >= 1e3:
        return f"{n_params / 1e3:.2f}K"
    else:
        return str(n_params)

def init_vjepa2(
    model_name: str="facebook/vjepa2-vitl-fpc64-256",
    dtype: torch.dtype=torch.float32,
):
    """
    Initializes the V-JEPA 2 model and its corresponding video processor.
    The model is frozen (parameters are not trainable) and set to evaluation mode.
    
    Model zoo:
    - https://huggingface.co/docs/transformers/model_doc/vjepa2
    - https://huggingface.co/collections/facebook/v-jepa-2
    """
    processor = AutoVideoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    model.eval()
    for p in model.parameters():
        p.require_grad = False
    print("model total parameters:", count_parameters(model))
    return model, processor