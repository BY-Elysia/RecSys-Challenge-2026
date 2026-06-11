from .doubao_api import DOUBAO_API_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    if lm_type == "doubao_api":
        return DOUBAO_API_MODEL()
    else:
        raise ValueError(f"Unsupported LM type: {lm_type}")
