# Proposal

## Why

The four attention modules in `compgen/models/attentions/` were written independently and have drifted: each repeats its own head-splitting math (`head_dim = hidden_dim // num_heads`, hardcoded Q=K=V width), its own `construct_mask`, its own projection layout, and its own scaling/dropout wiring. That means the head count and head dimension are effectively fixed by `hidden_dim`, and any new architecture copies the same boilerplate. The thesis now needs a *library of composable attention building blocks* — a softmax mixer, a linear mixer, and the legacy high-order variants — that all share one configuration surface (heads, QK/V dimensions, dropout, masking) so many models can be assembled from the same parts.

## What Changes

- **Introduce a shared attention base/interface.** A single base module owns the common configuration and mechanics every attention needs: `hidden_dim`, `num_heads`, resolved QK/V head dimensions, scaler, dropout, dtype/device, and mask construction. Each architecture implements only its score function and value aggregation.
- **Make heads and dimensions independently configurable — for every architecture, not just softmax.** `num_heads` stays explicit; QK and V head dimensions are independently settable (e.g. `qk_head_dim`, `v_head_dim`, or `qk_dim`/`v_dim`), with a default that makes QK and V share the same head dimension when only one is given. This becomes the shared base's contract and therefore applies to `linear` and the migrated legacy variants too.
- **Rename `standard` to `softmax`. BREAKING.** The registry key and module become `softmax` (the softmax-scaled dot-product mixer). `standard` is no longer a selectable name; the two notebooks, `README.md`, `AGENTS.md`, and the `model-layer` spec are updated accordingly.
- **Add a `linear` attention.** The same architecture as `softmax` (same projection, scaling, masking, multi-head split) with the final softmax normalization removed — the attention weights are the scaled scores directly.
- **Migrate all existing variants onto the shared base.** `softmax`, `linear`, `strassen`, `triangular`, and `third_order` all build on the shared interface and expose the configurable head/dimension surface. The three high-order variants are **preserved, registered, and retained** — refactored onto the base, not deleted and not sidelined — since they may be useful in future work.
- **Registry and docs reflect the new set.** `ATTENTION_CLASSES` exposes `softmax`, `linear`, `strassen`, `triangular`, `third_order`; `README.md`/`AGENTS.md` describe the base + building blocks and the head/dimension config.

## Capabilities

### New Capabilities

- *(none — this refines how the existing model layer is organized and configured rather than introducing a separate capability)*

### Modified Capabilities

- `model-layer`: the architecture registry's name set and surface change (add `softmax`/`linear`, drop `standard`, retain the high-order variants), and a new requirement that all attention architectures share a common configuration interface with independently configurable head count and QK/V dimensions.

## Impact

- **Code**: `compgen/models/attentions/` — new shared base; `standard.py` renamed to a `softmax` module; new `linear` module; `strassen.py`, `triangular.py`, `third_order.py` refactored onto the base; `__init__.py` registry updated. `compgen/models/encoder.py` (and therefore `Match3Model`) gains the head/dimension configuration pass-through.
- **API**: **BREAKING** — `attention_type="standard"` no longer resolves; callers use `"softmax"`. Attention constructors gain independent QK/V dimension parameters with a same-dimension default. `Match3Model`'s `forward` output contract and the batch input contract are unchanged.
- **Docs/notebooks**: `notebooks/match3_depth_vs_strassen.ipynb` and `notebooks/match3_strassen_vs_standard_own_stack.ipynb` update `"standard"` → `"softmax"`; `README.md` and `AGENTS.md` updated.
- **Behavior**: `softmax` and the migrated legacy variants must remain numerically equivalent to before under their existing configurations (default head/dimension settings reproduce current results); `linear` is new.
- **Not touched**: experiment suites (`compgen/experiments/`) remain self-contained and unmodified; the dataset layer is unaffected.
