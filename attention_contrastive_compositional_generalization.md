---
title: Attention Contrastive Regularization for Compositional Generalization
aliases:
  - Shared-Term Attention Contrastive Learning
  - Fuzzy Logic Transformer Experiment
tags:
  - transformers
  - attention
  - compositional-generalization
  - contrastive-learning
  - fuzzy-logic
  - research
  - attention-as-hypernetwork
status: idea
created: 2026-08-31
---

# Attention Contrastive Regularization for Compositional Generalization

## Core Idea

The goal is to improve **compositional generalization** in a small Transformer by adding a contrastive inductive bias directly to an attention-derived representation.

The first experiment will use the fuzzy-logic task from **Attention as a Hypernetwork** and keep the original Transformer architecture and synthetic data-generation structure as close as possible to the paper.

The proposed training objective is:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{task}
+
\lambda_{con}\mathcal{L}_{contrastive}
$$

For the fuzzy-logic regression task:

$$
\mathcal{L}_{task}
=
\operatorname{MSE}(\hat y,y)
$$

The only methodological change in the first experiment should be the addition of the attention contrastive loss.

---

## Research Question

> Does encouraging attention representations to be similar for fuzzy-logic tasks that share at least one constituent term improve generalization to unseen combinations of previously encountered terms?

The focus is on **compositional OOD generalization**, not merely better fitting of the training distribution.

---

## Motivation

A fuzzy-logic task contains two constituent terms:

$$
C_i = \{T_{i1},T_{i2}\}
$$

If two tasks share one constituent, such as:

$$
\{T_1,T_2\}
$$

and

$$
\{T_1,T_7\},
$$

they contain the reusable component $T_1$.

The proposed bias encourages their attention representations to be similar.

The intuition is that the Transformer may learn reusable internal representations of constituents and then recombine them when it encounters a new composition.

---

# Experimental Setup

## Paper / Baseline

Reference:

**Attention as a Hypernetwork**  
Simon Schug et al., ICLR 2025

Official repository:

https://github.com/smonsays/hypernetwork-attention

The first experiment should reuse the original fuzzy-logic implementation rather than creating a new independent version.

---

## Transformer Architecture

Use the fuzzy-logic Transformer configuration from the paper/repository:

| Parameter | Value |
|---|---:|
| Transformer layers | 2 |
| Embedding dimension | 128 |
| Attention heads | 8 |
| Q/K dimension per head | 16 |
| Value dimension per head | 16 |
| MLP dimension | 256 |
| Dropout | 0 |
| Sequence length | 16 |
| Attention | Standard softmax attention |
| Normalization | LayerNorm |
| MLP activation | GeLU |

Keep the paper's:

- pre-normalization structure
- residual connections
- relative positional encoding
- causal/non-causal configuration
- optimizer
- learning-rate schedule
- batch size
- training procedure

unchanged for the controlled baseline comparison.

The main Transformer block is conceptually:

$$
Z =
\operatorname{MHA}(\operatorname{LayerNorm}(X)) + X
$$

$$
Y =
\operatorname{FFN}(\operatorname{LayerNorm}(Z)) + Z
$$

---

# Fuzzy-Logic Dataset

## Task Structure

Use the same synthetic task family as:

`logic_4var_2term`

Parameters:

$$
\text{num\_variables}=4
$$

$$
\text{num\_terms}=2
$$

Each input is:

$$
x=[x_1,x_2,x_3,x_4]
$$

with components sampled from the same distribution as the original implementation, typically:

$$
x_j \sim U(0,1)
$$

There are:

$$
2^4=16
$$

possible conjunction terms.

Each term chooses either:

$$
x_j
$$

or

$$
1-x_j
$$

for every variable.

The fuzzy AND is based on `min`, so a term has the form:

$$
T(x)=\min(v_1,v_2,v_3,v_4)
$$

where:

$$
v_j \in \{x_j,1-x_j\}.
$$

A task/function contains two distinct terms:

$$
C=\{T_a,T_b\}
$$

combined with fuzzy OR:

$$
f_C(x)=\max(T_a(x),T_b(x)).
$$

---

## Same Dataset Structure, Not Same Data Points

The experiment should reproduce the **same synthetic generation process**, not necessarily the authors' exact floating-point samples.

This means preserving:

- 4 variables
- 16 possible conjunction terms
- 2 terms per fuzzy function
- fuzzy AND/OR definitions
- train/test/OOD split logic
- input/output sequence format
- final-query masking
- evaluation protocol

The numerical $x$ values can be freshly sampled.

For a controlled baseline-vs-method comparison, use identical seeds and task splits wherever possible.

---

# Sequence Construction

For one fuzzy function $C$, sample:

$$
x_1,\dots,x_N
$$

with:

$$
N=16.
$$

Compute:

$$
y_t=f_C(x_t).
$$

Each token follows the original representation, conceptually:

$$
[x_t;y_t].
$$

Since $x_t$ has four dimensions, each token has dimension:

$$
4+1=5.
$$

For the final query token, hide its target:

$$
[x_N;0].
$$

The Transformer predicts:

$$
y_N.
$$

The constituent term IDs must **not** be exposed to the Transformer as input features.

---

# Positive Pair Definition

For samples $i$ and $j$:

$$
C_i=\{T_{i1},T_{i2}\}
$$

$$
C_j=\{T_{j1},T_{j2}\}
$$

define them as a **positive pair** if they share at least one constituent term:

$$
C_i\cap C_j\neq\varnothing.
$$

Equivalently:

$$
T_{i1}=T_{j1}
\lor
T_{i1}=T_{j2}
\lor
T_{i2}=T_{j1}
\lor
T_{i2}=T_{j2}.
$$

### Examples

| Sample A | Sample B | Relation |
|---|---|---|
| `{T1,T2}` | `{T1,T5}` | Positive |
| `{T1,T2}` | `{T7,T2}` | Positive |
| `{T1,T2}` | `{T2,T1}` | Positive |
| `{T1,T2}` | `{T7,T8}` | Negative |

The pair of terms is treated as unordered.

---

# Attention Representation

Do **not** contrast the full attention matrix.

Use only the final Transformer layer.

Let $q$ be the final query-token position.

For each of the 8 heads, take the **post-softmax self-attention probability**:

$$
A_i^{(L,h)}[q,q].
$$

Construct:

$$
z_i=
[
A_i^{(L,1)}[q,q],
A_i^{(L,2)}[q,q],
\dots,
A_i^{(L,8)}[q,q]
].
$$

Therefore:

$$
z_i\in\mathbb{R}^{8}.
$$

For batch size $B$:

$$
Z\in\mathbb{R}^{B\times8}.
$$

Important:

- no projection MLP initially
- no hidden-state contrastive loss
- no full attention-map comparison
- do not detach $z_i$
- gradients from the contrastive loss must flow into the attention mechanism

---

# Contrastive Loss

Normalize each attention code:

$$
\bar z_i=
\frac{z_i}{\|z_i\|_2}.
$$

Compute pairwise cosine similarity:

$$
s(i,j)=\bar z_i^T\bar z_j.
$$

For anchor $i$, define:

$$
P(i)=
\{j\neq i:C_i\cap C_j\neq\varnothing\}.
$$

Use a supervised contrastive / InfoNCE-style objective:

$$
\mathcal{L}_i
=
-
\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log
\frac{
\exp(s(i,p)/\tau)
}{
\sum_{a\neq i}
\exp(s(i,a)/\tau)
}.
$$

Average over anchors with at least one positive:

$$
\mathcal{L}_{con}
=
\frac{1}{|V|}
\sum_{i\in V}
\mathcal{L}_i.
$$

Final objective:

$$
\boxed{
\mathcal{L}_{total}
=
\operatorname{MSE}(\hat y,y)
+
\lambda_{con}\mathcal{L}_{con}
}
$$

Initial hyperparameters:

$$
\lambda_{con}=0.05
$$

$$
\tau=0.1.
$$

These should remain configurable.

---

# First Experiment

Keep the first step deliberately simple.

## Experiment A — Baseline

Use the original paper-style setup.

Loss:

$$
\mathcal{L}=\operatorname{MSE}.
$$

## Experiment B — Proposed Method

Use exactly the same:

- Transformer architecture
- initialization procedure
- dataset generator
- train/OOD split
- sequence length
- batch size
- optimizer
- LR schedule
- training duration
- evaluation

Change only the loss:

$$
\mathcal{L}
=
\operatorname{MSE}
+
0.05\mathcal{L}_{con}.
$$

with:

$$
\tau=0.1.
$$

---

# What Not to Add Yet

For the first experiment, avoid unnecessary complexity.

Do not add:

- custom contrastive batch samplers
- projection heads
- weighted positive relationships
- triplet losses
- head selection
- multi-layer contrastive representations
- extra anti-collapse regularizers
- t-SNE analysis
- large hyperparameter sweeps
- additional model sizes
- unit-test infrastructure specifically for this experiment
- complicated diagnostic pipelines

Improve the experiment step by step only after obtaining the first baseline comparison.

---

# Logging

For now, log only the essentials:

- task MSE
- contrastive loss
- total loss
- ID $R^2$
- compositional OOD $R^2$
- average number of positive samples per anchor

---

# Evaluation

The main metric is **compositional OOD $R^2$**.

The coefficient of determination is:

$$
R^2
=
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}.
$$

Interpretation:

| $R^2$ | Meaning |
|---:|---|
| 1 | Perfect prediction |
| 0 | Equivalent to predicting the target mean |
| < 0 | Worse than predicting the mean |

**Higher $R^2$ is better.**

For MSE:

**Lower MSE is better.**

The desired result is:

$$
R^2_{OOD,contrastive}
>
R^2_{OOD,baseline}
$$

while ideally maintaining comparable ID performance.

---

# Main Hypothesis

The proposed inductive bias is:

> If two fuzzy-logic tasks contain the same reusable constituent term, their attention-level task representations should share structure.

For example:

$$
T_1\lor T_2
$$

and:

$$
T_1\lor T_7
$$

share $T_1$, so their attention codes are encouraged to be closer.

The hoped-for effect is that the Transformer learns reusable constituent representations that can be recombined for unseen compositions.

---

# First Implementation Goal

The first implementation should answer only one question:

> Does adding shared-term attention contrastive regularization improve compositional OOD performance compared with the original fuzzy-logic Transformer baseline?

If the result is promising, possible later extensions include:

- exact-pair vs shared-term positives
- graded similarity for 0/1/2 shared terms
- context-length sweep: 16, 32, 64, 96
- different contrastive weights and temperatures
- different layers
- per-head analysis
- contrastive-aware batch sampling
- attention-representation visualization

These are follow-up experiments, not part of the first step.
