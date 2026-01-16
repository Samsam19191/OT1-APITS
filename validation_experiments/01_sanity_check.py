import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import load_model_and_tokenizer, print_kv_stats
from src.prompts import SYSTEM_PROMPT, USER_PROMPT, SCHEMA_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_id)

    user_prompt = USER_PROMPT.format(question="Get all users older than 30.")
    schema = SCHEMA_PROMPT.format(schema="""Table users:
- id: integer, primary key
- name: text
- age: integer
- email: text""")
    
    prompt = SYSTEM_PROMPT + "\n" + schema + user_prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nRunning forward pass on prompt ({inputs.input_ids.shape[1]} tokens)...")
    with torch.no_grad():
        # Use model() for prefill (KV extraction), not generate()
        outputs = model(**inputs, use_cache=True)
    
    past_key_values = outputs.past_key_values
    
    print("\n[CHECK 1] KV Cache Extraction")
    if past_key_values is None:
        print("❌ FAIL: past_key_values is None")
        exit(1)
    else:
        print("✅ SUCCESS: past_key_values obtained")

    print_kv_stats(past_key_values, "Initial KV")
    
    # Verify we didn't generate tokens (output logits should match input length)
    logits_shape = outputs.logits.shape
    input_len = inputs.input_ids.shape[1]
    
    print(f"\n[CHECK 2] Shape Validation")
    print(f"Input Length: {input_len}")
    print(f"Logits Shape: {logits_shape}")
    
    if logits_shape[1] != input_len:
         print(f"❌ FAIL: Logits length {logits_shape[1]} != Input length {input_len}. Did it generate?")
         exit(1)
    else:
         print("✅ SUCCESS: Logits shape matches input length (no generation).")

    # CHECK 3: Actual SQL Generation
    print(f"\n[CHECK 3] SQL Generation")
    
    # Create stop sequence for end of SQL codeblock
    stop_token_ids = tokenizer.encode("\n```\n", add_special_tokens=False)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids[-1]  # Stop at ``` closing
        )
    
    # Decode only the new tokens (skip the prompt)
    new_tokens = generated_ids[0, input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract SQL from codeblock
    print(f"Full Response:\n{response}\n")
    
    if "```sql" in response:
        sql = response.split("```sql")[1].split("```")[0].strip()
        print(f"✅ Extracted SQL:\n   {sql}")
    else:
        print("⚠️  WARNING: No ```sql codeblock found in response")
        sql = response.strip()
        print(f"Raw output: {sql}")

if __name__ == "__main__":
    main()
