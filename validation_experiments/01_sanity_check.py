import argparse
import torch
from utils import load_model_and_tokenizer, print_kv_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_id)

    prompt = "Hello, this is a test of the KV cache system."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nRunning forward pass on: '{prompt}'")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    past_key_values = outputs.past_key_values
    
    print("\n[CHECK 1] extraction")
    if past_key_values is None:
        print("FAIL: past_key_values is None")
        exit(1)
    else:
        print("SUCCESS: past_key_values obtained")

    print_kv_stats(past_key_values, "Initial KV")
    
    # Verify we didn't generate tokens (output logits should match input length)
    logits_shape = outputs.logits.shape
    input_len = inputs.input_ids.shape[1]
    
    print(f"\n[CHECK 2] shape validation")
    print(f"Input Length: {input_len}")
    print(f"Logits Shape: {logits_shape}")
    
    if logits_shape[1] != input_len:
         print(f"FAIL: Logits length {logits_shape[1]} != Input length {input_len}. Did it generate?")
         exit(1)
    else:
         print("SUCCESS: Logits shape matches input length (no generation).")

if __name__ == "__main__":
    main()
