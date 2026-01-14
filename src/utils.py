import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import os

CONFIRMATION_BOUNDARIES = {' ', '\n', '\t', '.', ',', ';', ':', '!', '?', '(', ')', '"', "'"}

def load_model_and_tokenizer(model_id):
    """
    Loads model and tokenizer efficiently.
    Uses bfloat16 if available, else float16 or float32.
    """
    print(f"Loading model: {model_id}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mac M1/M2/M3 support (MPS)
    if torch.backends.mps.is_available():
        device = "mps"
        
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        dtype = torch.float16 # float16 on CPU can be slow or unsupported for some ops
        
    print(f"Using device: {device}, dtype: {dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device != "mps" else None, # accelerate handles device_map often better, but for MPS manual move is safer sometimes. 
            # Simple approach: let accelerate handle it if CUDA, else manual.
            # actually for simple scripts, explicit .to(device) is safer than device_map="auto" which might split across CPU/GPU unexpectedly if limited vram.
            # But let's try standard loading first.
            trust_remote_code=True,
            # quantization_config=GPTQConfig(bits=4, use_exllama=False)
        )
        if device == "mps":
            model = model.to(device)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    return model, tokenizer, device

def find_last_boundary(text: str) -> int:
    """
    Find the index after the last confirmation boundary in text.
    
    Returns the position up to which tokens can be safely confirmed.
    """
    last_boundary_idx = -1
    for i, char in enumerate(text):
        if char in CONFIRMATION_BOUNDARIES:
            last_boundary_idx = i
    return last_boundary_idx + 1 if last_boundary_idx >= 0 else 0