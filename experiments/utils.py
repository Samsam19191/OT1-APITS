import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

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
        dtype = torch.float32 # float16 on CPU can be slow or unsupported for some ops
        
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
            trust_remote_code=True
        )
        if device == "mps":
            model = model.to(device)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    return model, tokenizer, device

def print_kv_stats(past_key_values, label="KV Stats"):
    """
    Prints shape and basic stats of the KV cache.
    Expects tuple of tuples (standard HF format) or DynamicCache.
    """
    print(f"--- {label} ---")
    if past_key_values is None:
        print("None")
        return

    # Handle DynamicCache (new HF style)
    if hasattr(past_key_values, "get_seq_length"):
        bg_len = past_key_values.get_seq_length()
        print(f"DynamicCache successfully detected. Seq Len: {bg_len}")
        # Try to inspect first layer
        if hasattr(past_key_values, "key_cache"):
             if len(past_key_values.key_cache) > 0:
                 k_shape = past_key_values.key_cache[0].shape
                 print(f"Layer 0 Key Shape: {k_shape}")
        elif hasattr(past_key_values, "keys"): # Some variants
             if len(past_key_values.keys) > 0:
                 k_shape = past_key_values.keys[0].shape
                 print(f"Layer 0 Key Shape: {k_shape}")
        else:
             # Fallback: try to access as list/tuple if possible, or just print string rep
             try:
                 print(f"Structure: {type(past_key_values)}")
                 # print(f"Dir: {dir(past_key_values)}") 
             except:
                 pass
        return

    # Handle legacy tuple of tuples
    num_layers = len(past_key_values)
    print(f"Num Layers: {num_layers}")
    if num_layers > 0:
        # (batch, num_heads, seq_len, head_dim) usually
        # keys are [0], values are [1]
        k_shape = past_key_values[0][0].shape
        v_shape = past_key_values[0][1].shape
        print(f"Layer 0 Key Shape: {k_shape}")
        print(f"Layer 0 Value Shape: {v_shape}")
