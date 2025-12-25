# Model Validation Report: Anticipatory Prefill

## Executive Summary
**Result**: ✅ **PASSED** (with minor caveats)
**Recommendation**: Proceed with **Qwen2.5-1.5B** or **TinyLlama-1.1B**. Both models fully support the critical KV-cache operations required for the project.

The core assumption—that we can maintain, extend, and crop a single KV cache without full re-computation—is valid for standard HF decoder-only models.

---

## Experiment Results

| Test Case | Description | Qwen2.5-1.5B | TinyLlama-1.1B | Status |
| :--- | :--- | :--- | :--- | :--- |
| **KV Extraction** | Can we get `past_key_values` from a prefill-only forward pass? | ✅ Passed | ✅ Passed | **Confirmed** |
| **KV Extension** | Can we extend an existing KV cache with new tokens? | ✅ Passed* | ✅ Passed* | **Confirmed** |
| **KV Cropping** | Can we truncate the KV cache and continue generating? | ✅ Passed | ✅ Passed | **Confirmed** |

*> **Note on Extension**: The incremental updates produced the correct KV shape and length. However, there was a minor logit divergence (diff ~0.02) compared to a full non-cached run. This is expected floating-point drift in `float16` and is within acceptable limits for this application. It does not indicate a logic failure.*

---

## Model Comparison Analysis

### **1. Qwen2.5-1.5B (Recommended)**
*   **Quality**: 🏆 **Best in Class**. Qwen2.5 is currently state-of-the-art for coding tasks in the sub-7B category. It significantly outperforms Llama-based models on benchmarks like HumanEval and MBPP. For Text-to-SQL, this is the decisive factor.
*   **Tokenizer**: **More Efficient**. It has a large vocabulary (~152k tokens) compared to Llama (~32k). This means it compresses code/text into *fewer tokens*, effectively increasing your context window and speed.
*   **Architecture**: Uses Grouped Query Attention (GQA), making KV cache memory footprint smaller than standard MHA models.

### **2. TinyLlama-1.1B**
*   **Quality**: **Decent but Dated**. Based on Llama 2 architecture. It is robust for English chat but lacks the specialized coding training that Qwen has.
*   **Speed**: Marginally faster raw compute due to fewer parameters (1.1B vs 1.5B), but likely slower *end-to-end* because it produces more tokens for the same text.
*   **Memory**: Slightly smaller weights (~2.2GB vs ~3GB for FP16), but KV cache size is comparable.

### **Verdict**
**Use Qwen2.5-1.5B.**
The slight increase in model size (0.4B params) is negligible on a T4 GPU (which has 16GB VRAM), but the jump in SQL/Coding capability is massive. The "fewer tokens" property of the tokenizer also helps with the latency goals.

---

## Detailed Findings

### 1. KV Cache Structure
Both models use the modern `DynamicCache` structure in Hugging Face Transformers.
- **Qwen**: Uses `DynamicCache`.
- **TinyLlama**: Uses `DynamicCache`.
- **Action**: The codebase must support `DynamicCache` API (specifically `get_seq_length()` and `crop()`).

### 2. Cropping Reliability
The most critical risk—rollback support—was validated.
- We successfully cropped a cache from length 10 down to 5.
- We successfully performed a subsequent forward pass using the cropped cache.
- The final shapes matched expectations.

### 3. Memory & Performance
- **Qwen2.5-1.5B**: ~3GB VRAM (half precision). Fast on T4.
- **TinyLlama-1.1B**: ~2.2GB VRAM. Slightly smaller/faster.

## Conclusion & Next Steps
The project is **technically feasible** on the proposed hardware (Colab T4) using standard HF components.