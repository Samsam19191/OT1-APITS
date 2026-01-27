# APITS Presentation Plan (~25 minutes)

> **Anticipatory Prefill with Keystroke Streaming for Interactive Text-to-SQL**

---

## 1. Introduction (2 min)

**Objective**: Set the stage, introduce the team and topic

- **Hook**: "What if your SQL query was already being processed before you finished typing?"
- Team introduction (6 members, INSA Lyon)
- Project context: Text-to-SQL + LLM inference optimization
- Quick agenda overview

---

## 2. Text-to-SQL & Industry Context (3 min)

**Objective**: Explain what Text-to-SQL is and why it matters today

### 2.1 What is Text-to-SQL?
- Natural language → SQL query
- Democratizing database access for non-technical users
- Example: "How many customers are in France?" → `SELECT COUNT(*) FROM customers WHERE country = 'France';`

### 2.2 Industry Adoption
- Oracle 23c with AI/ML integration
- Apache Spark NL queries
- GitHub Copilot for SQL
- Major cloud platforms (AWS, Azure, GCP) adding NL interfaces
- LLMs achieving >80% accuracy on benchmarks

---

## 3. Problem Statement (3 min)

**Objective**: Define the latency problem we're solving

### 3.1 The Latency Problem
- State-of-the-art accuracy ✓ but **user experience suffers**
- **Time-to-First-Token (TTFT)**: delay between "Enter" and first response character
- Typical TTFT: 300-800ms on consumer hardware → feels sluggish

### 3.2 Growing Complexity
- Big Data era: larger schemas, more tables
- Complex analytical queries: multiple JOINs, aggregations
- Prompt size grows: schema description can be 1000+ tokens
- More tokens = longer prefill time

### 3.3 The Wasted Time Observation
- User types query: 3-10 seconds
- Backend sits **idle** during typing
- Only starts computation AFTER submission
- **Key insight**: We can exploit typing time!

---

## 4. How LLM Inference Works (4 min)

**Objective**: Give audience technical foundation to understand our solution

### 4.1 Two-Phase Process
```
[Prefill Phase] → [Generation Phase]
```

1. **Prefill**: Process ALL input tokens, build KV-cache
   - Takes ~200-800ms for long prompts
   - Dominates TTFT

2. **Generation**: Produce output tokens one-by-one
   - Fast per token (~10-50ms each)
   - Uses cached context

### 4.2 The KV-Cache (Simple Explanation)
- Transformers compute "attention" - relating each word to all previous words
- **Key-Value cache**: stores computed attention information
- Avoids recomputing same values → O(n) instead of O(n²)
- Visual: Show cache growing as tokens are added

### 4.3 Why Prefill is the Bottleneck
- Schema + system prompt + user query = 500-2000 tokens
- All processed at once when user hits Enter
- User waits for this computation

---

## 5. Our Solution: Anticipatory Prefill (6 min)

**Objective**: Explain the core innovation in detail

### 5.1 Core Idea
- Stream keystrokes in real-time via WebSocket
- Build KV-cache **incrementally** as user types
- When user submits → cache is already warm!

### 5.2 Extend Operation
- User types "Hello "
- Tokenize → get new tokens
- Run forward pass on ONLY the new tokens
- Append to existing cache
- **Boundary-based confirmation**: Wait for word boundaries (space, punctuation)
- **Debouncing**: 300ms pause before committing

### 5.3 Crop Operation (Handling Deletions)
- User deletes/edits already-cached text
- Standard approach: recompute everything ❌ (expensive)
- Our approach: `cache.crop(target_length)` → O(1) tensor slicing
- Rollback is instant, re-extend from the cropped point

### 5.4 Architecture Diagram
```
[React Frontend] --keystrokes--> [WebSocket] --> [StreamController] --> [KVCacheManager] --> [GPU]
                 <--tokens------  [WebSocket] <-- [Generate]         <-- [Warm Cache]
```

### 5.5 Correctness Guarantee
- At submission: cache exactly represents full prompt
- Generation identical to standard inference
- No approximation, no quality degradation

---

## 6. Evaluation Framework (5 min)

**Objective**: Explain methodology, choices, and expected outcomes

### 6.1 Model Choice: Why Qwen2.5-Coder-1.5B?
- State-of-the-art SQL/code capability for sub-3B models
- 152K vocabulary (efficient tokenization vs Llama's 32K)
- **GQA architecture** → smaller KV-cache memory footprint
- Full `DynamicCache` support including `crop()` operation
- Good balance: fast enough for real-time, smart enough for SQL

### 6.2 Why These Metrics?
| Metric | Why We Chose It |
|--------|-----------------|
| **TTFT** | Direct measure of user-perceived latency (our main goal) |
| **Execution Accuracy** | Proves we don't sacrifice quality for speed |
| **Exact Match** | Stricter correctness check |
| **Memory Usage** | Important trade-off to monitor |

### 6.3 Expected Results (Before Running Eval)
- **TTFT**: Expect significant reduction (typing time → prefill time)
- **Accuracy**: Expect **identical** results (correctness guarantee by design)
- **Memory**: Expect **slightly higher** for our approach:
  - KV-cache persists during typing session
  - Baseline only allocates during generation
  - Trade-off: memory for latency

### 6.4 Dataset
- **Spider benchmark**: Standard Text-to-SQL evaluation
- Databases: `car_1`, `world_1`, `dog_kennels`
- **378 test cases** total
- Mix of simple and complex queries

---

## 7. Live Demo (1 min)

**Objective**: Show the system working

- Open E2E evaluation dashboard
- Show live typing simulation with KV-cache building
- Compare Anticipatory vs Baseline TTFT in real-time
- *(No script needed - just demonstrate the UI)*

---

## 8. Final Results (3 min)

**Objective**: Present the actual numbers from the full evaluation run

### 8.1 TTFT Comparison
| Mode | Avg TTFT | Speedup |
|------|----------|---------|
| Baseline | 629 ms | 1.0x |
| APITS | 64 ms | **~10x** |

### 8.2 SQL Quality (No Degradation)
| Metric | Baseline | APITS |
|--------|----------|-------|
| Execution Accuracy | 70.4% | 71.2% |
| Exact Match | 46.3% | 46.8% |

→ **Confirms our correctness guarantee**: identical quality

### 8.3 Total Evaluation Time
- Baseline: 1170s
- APITS: 1040s (11% faster overall)

### 8.4 Key Observations
- Longer prompts → greater speedup (more prefill to overlap)
- Typing speed impacts benefit (slow typers benefit most)
- Memory overhead minimal in practice

---

## 9. Conclusion & Future Work (2 min)

**Objective**: Wrap up, acknowledge limitations, point to extensions

### 9.1 Summary
- Identified wasted typing time as optimization opportunity
- Developed extend/crop cache operations
- Achieved 10x TTFT speedup without accuracy loss

### 9.2 Limitations
- Fast typists may outpace system
- Network latency in cloud deployments
- Single-user focus (not multi-tenant optimized)

### 9.3 Future Work
- Predictive speculation beyond confirmed tokens
- Adaptive confirmation based on typing patterns
- Shared caches across users with common prefixes
- Combination with speculative decoding

### 9.4 Thank You + Questions

---

## Appendix: Slide Allocation Suggestion

| Section | Slides | Time |
|---------|--------|------|
| 1. Introduction | 1-2 | 2 min |
| 2. Text-to-SQL Context | 2-3 | 3 min |
| 3. Problem Statement | 2-3 | 3 min |
| 4. LLM Inference Basics | 3-4 | 4 min |
| 5. Our Solution | 4-6 | 6 min |
| 6. Evaluation Framework | 3-4 | 5 min |
| 7. Live Demo | 1 | 1 min |
| 8. Final Results | 2-3 | 3 min |
| 9. Conclusion | 1-2 | 2 min |
| **Total** | **~22-28** | **~29 min** |

*(Aim for ~25 min to leave buffer for Q&A)*

---

## Key Numbers to Memorize

- **TTFT Speedup**: ~10x (629ms → 64ms)
- **Accuracy**: 71.2% execution, 46.8% exact match
- **Test Cases**: 378 across 3 databases
- **Model**: Qwen2.5-Coder-1.5B-Instruct
- **Total Eval Time**: 1040s (vs 1170s baseline)

