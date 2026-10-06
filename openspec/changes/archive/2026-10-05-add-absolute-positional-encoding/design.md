# Design

## Context

See `proposal.md` — Why. Current state that shapes the approach:

- Position is injected today only by `RelativePositionBias` in `compgen/models/position.py`: an `nn.Embedding(num_buckets, num_heads)` table, T5-style unidirectional bucketing of `query − key` distances, returning `(1, num_heads, N, N)` added to attention scores. It is owned by `EncoderLayer` (`use_relative_bias`, `relative_bias_num_buckets`, `relative_bias_max_distance`) and configured through `FuzzyLogicModel`, the experiment `ExperimentConfig`, and `run.build_model`.
- `FuzzyLogicEmbedding` (`compgen/models/embeddings/fuzzy_logic.py`) is only `Linear(input_dim, hidden_dim)` plus dropout — it carries no position signal.
- The Match3 embedding already bakes position into hand-crafted features (`[position, value, value², 1]`) and does not use the relative bias, so it is out of scope.
- Attention (`softmax`, `linear`) accepts an optional additive `score_bias`; that is a generic interface, not relative-position-specific.

## Goals / Non-Goals

**Goals:**
- Introduce one fixed, deterministic absolute positional encoding and add it to the fuzzy-logic input embedding.
- Remove the relative-position bias subsystem and every option/wiring that exposes it.
- Keep a documented record of the removed idea for possible future re-testing.

**Non-Goals:**
- Changing attention architectures, the shared registry, the feed-forward block, the regression head, or the training/tracking layers.
- Changing the Match3 model or its embeddings.
- Making the positional encoding learnable, or reintroducing any relative/learned position scheme now.
- Changing dataset generation or the `seq_len` contract.

## Decisions

### D1. Fixed sinusoidal encoding, computed on the fly
Use the standard Vaswani sinusoidal formulation: for position `p` and channel `i`, `PE[p, 2i] = sin(p / 10000^(2i/hidden))` and `PE[p, 2i+1] = cos(p / 10000^(2i/hidden))`, shape `(length, hidden)`.

- **Why**: the user asked for a *fixed* positional encoding added to the embedding matrix; sinusoidal is parameter-free, deterministic, dtype/device-castable, and defined for any length, so it needs no `seq_len` bound and adds nothing to the checkpoint.
- **Alternatives considered**: a learned `nn.Embedding(seq_len, hidden)` (rejected — not fixed, and it caps sequence length and adds parameters); keeping relative bias alongside (rejected — explicit ask to remove).
- **Placement**: put the encoding in `compgen/models/position.py`, replacing the relative-bias content, so the model layer still has one small reusable position block. `FuzzyLogicEmbedding` adds it: `dropout(input_proj(x) + pe[:length])`.
- **Caching**: compute in `forward` from `x.shape[1]`; it is cheap and length-agnostic. No buffer is registered (avoids a fixed-length assumption and checkpoint churn). Cast to the module's `dtype`/`device`.

### D2. Unconditional in the fuzzy-logic embedding
The fuzzy-logic embedding always adds the encoding; no new flag is introduced.

- **Why**: after removing relative bias this is the model's only position signal, so it should be on by default and not regress to a position-blind model by omission. Keeping a toggle would preserve the relative-bias-style configuration surface the change is meant to delete.
- **Alternative considered**: a `use_positional_encoding` flag (rejected as unnecessary configuration for a single internal model).

### D3. Remove relative bias and its configuration entirely
Delete `RelativePositionBias` and `relative_position_buckets`; drop `use_relative_bias`, `relative_bias_num_buckets`, `relative_bias_max_distance` from `EncoderLayer` and `FuzzyLogicModel`, from `ExperimentConfig`, and from `run.build_model`. `EncoderLayer.forward` keeps its optional `score_bias` argument but no longer derives one automatically.

- **Why**: the generic additive-bias instrument stays available (architecture-neutral), while the relative-position owner disappears. This keeps `EncoderLayer` usable and avoids touching attention.
- **Alternatives considered**: leave dead options in place (rejected — the request is to remove the functionality "at all"); remove `score_bias` too (rejected — it is a general attention interface used by the attention contract requirement, not relative-bias-specific).

### D4. Preserve the idea as a tracked note under `results/`
Add `results/relative_position_bias.md` describing what the removed block did and how to re-test it, and add a `.gitignore` negation so this single markdown is tracked even though `results/` is otherwise ignored.

- **Why**: the user asked for the idea to live "in results" as a markdown. `results/` is gitignored, so without an exception the note would not be preserved in version control.
- **Alternative considered**: put it in `notes/` (rejected — the user explicitly chose `results/`; this is flagged as a trade-off below).

## Risks / Trade-offs

- **Behavior change is breaking and intentional** → The fuzzy-logic model no longer reproduces the reference "Attention as a Hypernetwork" architecture (its parameter count and per-layer position behavior change). Mark it in `proposal.md` as BREAKING; capture the old design in the `results/` note.
- **Absolute PE is added once at the input, relative bias acted every layer** → position signal may matter less in deep/long stack; this is the accepted trade-off of the refactor and the reason the note records how to re-test the relative variant.
- **`results/` is gitignored** → a negation pattern is required, and the task must verify the note is actually tracked (`git check-ignore`/`git status`) rather than assuming.
- **Docstring drift** → several docstrings describe relative-position bias (`encoder.py`, `tasks/fuzzy_logic.py`, `attentions/softmax.py`, `attentions/linear.py`, `position.py`) and `AGENTS.md` names the path's purpose. All must be updated together or docs misdescribe the code.
- **Interaction with unrelated local edits** → `ExperimentConfig` currently has separate uncommitted changes to head/FFN widths; this change must only touch the relative-bias fields there.

## Migration Plan

1. Replace `compgen/models/position.py` with the fixed encoding block.
2. Add the encoding in `FuzzyLogicEmbedding`.
3. Strip relative-bias options/wiring from `EncoderLayer`, `FuzzyLogicModel`, `config.py`, `run.py`; update docstrings and `AGENTS.md`.
4. Add the `results/` note and the `.gitignore` exception.
5. Verify: package imports from the repo root, the fuzzy-logic model constructs and forwards, the four-variant runner starts, no `use_relative_bias`/`relative_bias` references remain, and the note is tracked.
6. Rollback: revert the change's commit; no external consumers depend on the removed options.

## Open Questions

None.
