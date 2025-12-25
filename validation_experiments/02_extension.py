import argparse
import torch
from utils import load_model_and_tokenizer, print_kv_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_id)
    
    # Define parts
    part1_text = "The quick brown fox"
    part2_text = " jumps over the lazy dog"
    full_text = part1_text + part2_text
    
    print(f"Part 1: '{part1_text}'")
    print(f"Part 2: '{part2_text}'")
    
    # 1. Full non-cached run (Reference)
    print("\n--- 1. Running Full Reference ---")
    inputs_full = tokenizer(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_ref = model(**inputs_full, use_cache=True)
    ref_kv = outputs_ref.past_key_values
    print_kv_stats(ref_kv, "Reference KV")
    
    # 2. Incremental run
    print("\n--- 2. Running Incremental Step 1 ---")
    inputs_p1 = tokenizer(part1_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_p1 = model(**inputs_p1, use_cache=True)
    kv_p1 = outputs_p1.past_key_values
    print_kv_stats(kv_p1, "KV Step 1")
    
    print("\n--- 3. Running Incremental Step 2 (Extension) ---")
    # Crucial: verify we can extend.
    # Note: tokenizer(part2_text) might add a space differently if not careful, 
    # but for this test we focus on mechanism, not perfect token matching quirks yet.
    # We strip special tokens if necessary to avoid double BOS.
    # Actually, simpler to just get full tokens and slice them to ensure identity.
    
    ids_full = inputs_full.input_ids
    len_p1 = inputs_p1.input_ids.shape[1]
    
    # Use exact tokens from reference to avoid tokenizer mismatch noise
    ids_p2 = ids_full[:, len_p1:]
    
    print(f"Step 1 Tokens: {inputs_p1.input_ids.tolist()}")
    print(f"Step 2 Tokens (to extend): {ids_p2.tolist()}")
    
    if ids_p2.shape[1] == 0:
        print("FAIL: Part 2 is empty due to tokenization overlap? Check split.")
        exit(1)

    with torch.no_grad():
        # Pass past_key_values=kv_p1
        outputs_p2 = model(input_ids=ids_p2, past_key_values=kv_p1, use_cache=True)
    
    kv_final = outputs_p2.past_key_values
    print_kv_stats(kv_final, "Final Incremental KV")
    
    # Validation
    # Check lengths
    if hasattr(kv_final, "get_seq_length"):
        len_ref = ref_kv.get_seq_length()
        len_final = kv_final.get_seq_length()
    else:
        len_ref = ref_kv[0][0].shape[2]
        len_final = kv_final[0][0].shape[2]
        
    print(f"\n[CHECK] Length Validation: Ref={len_ref}, Incremental={len_final}")
    if len_ref != len_final:
        print(f"FAIL: Mismatch in final KV length.")
        exit(1)
    
    # Check content (approximate check on last logits or some KV values)
    # Checking last logit match
    ref_last_logit = outputs_ref.logits[0, -1, :]
    inc_last_logit = outputs_p2.logits[0, -1, :]
    
    diff = (ref_last_logit - inc_last_logit).abs().max()
    print(f"Max Logit Difference: {diff.item()}")
    
    if diff.item() > 1e-3: # loose tolerance for float16/bf16
        print("FAIL: Logits diverged significantly.")
        exit(1)
    
    print("SUCCESS: Incremental prefill matches reference.")

if __name__ == "__main__":
    main()
