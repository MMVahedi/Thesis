---

tags:

* transformers
* chain-of-thought
* rnn
* theory
* neurips2025

---

# Compositional Reasoning with Transformers, RNNs, and Chain of Thought

## Links

* Paper (OpenReview PDF): https://openreview.net/pdf?id=nUZaI7aRb2
* OpenReview Discussion: https://openreview.net/forum?id=nUZaI7aRb2

---

# Metadata

* Venue: NeurIPS 2025
* Type: Theoretical / Expressivity Analysis
* Experiments: ❌ None
* Training Analysis: ❌ None
* Optimization Analysis: ❌ None
* Dataset / Benchmark: CRQ (Compositional Reasoning Questions)

---

# TL;DR

The paper introduces a synthetic reasoning task called **Compositional Reasoning Questions (CRQs)** and uses it to compare three approaches to reasoning:

* Deep Transformers
* RNNs
* Transformers with Chain-of-Thought

The central claim is:

> All three architectures can solve the same reasoning problem, but each architecture pays for reasoning using a different computational resource.

---

# Main Question

Where does reasoning happen?

When a model solves a multi-step reasoning problem, the computation must be stored somewhere.

The paper studies three possibilities:

```text
Transformer
    → Layers

RNN
    → Hidden State

Chain of Thought
    → Generated Tokens
```

---

# CRQ (Compositional Reasoning Questions)

CRQ is a synthetic task designed to capture hierarchical reasoning.

A problem is represented as a tree:

```text
Root Question
     |
 Subquestions
     |
 Smaller Subquestions
     |
 Leaf Facts
```

The answer is obtained by recursively combining answers from lower levels.

The authors show that CRQ captures Boolean Formula Evaluation, a canonical NC¹-complete problem.

Because of this, CRQ serves as a clean theoretical benchmark for studying compositional reasoning.

---

# Core Insight

Reasoning is not free.

A model must spend some computational resource to solve a compositional reasoning problem.

Different architectures spend different resources:

### Transformers

Spend depth.

More reasoning steps require more layers.

### RNNs

Spend memory.

Intermediate results are stored in the hidden state.

### Chain of Thought

Spend inference-time computation.

Intermediate results are stored in generated tokens.

---

# Main Results

### Deep Transformers

Can solve CRQs.

Reasoning progresses layer-by-layer through the tree.

More reasoning depth requires more transformer depth.

---

### RNNs

Can solve CRQs.

Reasoning is implemented through hidden-state memory.

A carefully chosen ordering allows logarithmic memory usage.

---

### Chain of Thought

Can solve CRQs using a constant-depth transformer.

The reasoning process is externalized into generated intermediate tokens.

---

# Main Conclusion

The paper does **not** identify a single best reasoning architecture.

Instead, it shows that:

* Transformers,
* RNNs,
* and Chain-of-Thought

are different ways of implementing the same reasoning process.

The difference is not whether reasoning happens.

The difference is where the computation is stored.

---

# What This Paper Does NOT Study

* Learning dynamics
* Optimization
* SGD convergence
* Generalization
* Scaling laws
* Real-world reasoning benchmarks
* Empirical performance

The paper is entirely about **expressive power**.

The results are existence proofs showing that certain architectures can represent CRQ solutions.

---

# Personal Takeaway

Most reasoning papers ask:

> Which architecture is stronger?

This paper asks a different question:

> If reasoning must happen, where is the computation stored?

Its answer is:

```text
Depth
Memory
Inference-Time Computation
```

These are three different currencies for paying the cost of reasoning.
