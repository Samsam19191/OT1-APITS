import argparse
import torch
from utils import load_model_and_tokenizer, print_kv_stats
from transformers.cache_utils import DynamicCache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_id)

    # 1. Create a large KV
    full_text = "One two three four five six seven eight nine ten"
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    original_kv = outputs.past_key_values
    
    print_kv_stats(original_kv, "Original KV")
    
    # Determine target length (let's crop to 5 tokens)
    target_length = 5
    print(f"\nAttempting to crop to length {target_length}...")
    
    # API Detection and Cropping
    if isinstance(original_kv, DynamicCache):
        print("Detected DynamicCache. Using crop()")
        # Note: crop() in DynamicCache usually updates in-place or returns?
        # HF source says: crop(self, maximum_length: int) -> None (In-place)
        try:
            original_kv.crop(target_length)
            cropped_kv = original_kv
        except Exception as e:
            print(f"FAIL: DynamicCache.crop raised exception: {e}")
            exit(1)
            
        # Verify length
        if original_kv.get_seq_length() != target_length:
            print(f"FAIL: DynamicCache length is {original_kv.get_seq_length()} after crop, expected {target_length}")
            exit(1)
            
    else:
        print("Detected Tuple KV (Legacy). Manually slicing.")
        # Tuple of (key, value) tensors
        # Shape: (batch, n_heads, seq_len, head_dim)
        # We need to slice dim 2
        try:
            cropped_kv = []
            for layer_idx, (k, v) in enumerate(original_kv):
                k_cropped = k[..., :target_length, :]
                v_cropped = v[..., :target_length, :]
                cropped_kv.append((k_cropped, v_cropped))
            cropped_kv = tuple(cropped_kv)
        except Exception as e:
             print(f"FAIL: Manual slicing error: {e}")
             exit(1)

        # Verify length
        curr_len = cropped_kv[0][0].shape[2]
        if curr_len != target_length:
            print(f"FAIL: Sliced length is {curr_len}, expected {target_length}")
            exit(1)
            
    print("SUCCESS: Cropping operation completed without error.")
    print_kv_stats(cropped_kv, "Cropped KV")
    
    # 2. Verify usability of cropped KV
    # We should be able to extend from this cropped state
    next_text = " eleven twelve"
    ids_next = tokenizer(next_text, return_tensors="pt").input_ids.to(device)
    
    print("\nAttempting to extend from cropped KV...")
    try:
        with torch.no_grad():
            outputs_new = model(input_ids=ids_next, past_key_values=cropped_kv, use_cache=True)
        new_kv = outputs_new.past_key_values
        print("SUCCESS: Forward pass with cropped KV succeeded.")
        print_kv_stats(new_kv, "Extended KV")
        
    except Exception as e:
        print(f"FAIL: Extended forward pass crashed: {e}")
        # Make sure to print full error for debugging
        import traceback
        traceback.print_exc()
        exit(1)

    # Final check: Does total length match?
    # target_length + new tokens
    expected_len = target_length + ids_next.shape[1]
    
    if hasattr(new_kv, "get_seq_length"):
        real_len = new_kv.get_seq_length()
    else:
        real_len = new_kv[0][0].shape[2]
        
    if real_len == expected_len:
        print(f"SUCCESS: Final length {real_len} matches expected {expected_len}.")
    else:
        print(f"FAIL: Final length {real_len} != expected {expected_len}.")
        exit(1)

if __name__ == "__main__":
    main()
