# Design

## Context

See `proposal.md` — Why. Current state that shapes the approach:

- `compgen/models/attentions/__init__.py` imports the five concrete attention classes, exposes them through the `ATTENTION_CLASSES` dict, and adds a `build_attention(name, *, hidden_dim, ...)` helper that looks the name up and constructs the module.
- `Match3Model` and `FuzzyLogicModel` both call `build_attention(attention_type, ...)` inside their layer factory and expose `attention_type: str` on their constructors; they also bake the architecture name into `self.name`.
- The fuzzy-logic contrastive runner (`compgen/runners/fuzzy_logic_contrastive/`) carries architecture names as strings in `ExperimentConfig.attention_types`, `Variant.attention_type`, the CLI (`--attention-types`), and the JSON sweep grids (`scripts/grids/*.json`). `scripts/sweep_fuzzy_logic.py` uses `dataclasses.replace`/`asdict` on the config and hashes the override values, so those values must stay JSON-serializable.
- The encoder layer already consumes a constructed attention module (see the `model-layer` spec); it needs no change. The attention classes themselves and `BaseAttention` need no change.
- Precedent: the project already removed the analogous `build_ffn` name-keyed feed-forward factory, keeping the two feed-forward classes as directly instantiated modules (archived change `2026-10-12-encoder-dropout-and-attention-self-init`).

## Goals / Non-Goals

**Goals:**
- Remove the attention name→class registry and the name→module builder so architecture selection is explicit at every call site.
- Change the task models to accept a caller-supplied `attention_factory: Callable[[], nn.Module]` and obtain one module per encoder layer from it.
- Keep the fuzzy-logic contrastive experiment, its CLI, and its JSON sweep grids working, with observable four-variant behavior unchanged.
- Keep every attention class and the encoder layer untouched.

**Non-Goals:**
- Changing attention math, the `BaseAttention` configuration interface, the encoder layer, or the loss/training/tracking layers.
- Removing the runner's human-readable architecture labels or its string-based CLI/grid. Only the *model layer* loses name-based selection.
- Introducing a compatibility shim for the old `attention_type=`/`build_attention`/`ATTENTION_CLASSES` surface.

## Decisions

### 1. Task models take a zero-arg attention factory
`Match3Model(attention_factory=..., ...)` and `FuzzyLogicModel(attention_factory=..., ...)` replace `attention_type: str`. Inside the existing `make_layer`/layer list comprehension, each layer calls `attention_factory()` once and passes the returned module to `EncoderLayer`. Because the factory now owns the attention shape, both models also drop their now-dead `qk_head_dim`/`v_head_dim` parameters (they were only forwarded into the removed builder); the caller's factory carries those dimensions.

- Why a factory rather than a single module: each encoder layer owns its own attention parameters, so a fresh module per layer is required unless layers are shared. A factory preserves the multi-layer and `share_layers` semantics exactly.
- Alternatives considered: a constructed module argument (forces sharing and changes `share_layers` semantics); accepting a class plus kwargs (the task model would still "initiate" attention and would need a signature contract). The factory keeps construction entirely on the caller's side.

### 2. The attention package keeps direct class re-exports, drops the dict and builder
`compgen/models/attentions/__init__.py` retains its per-class imports (so `from compgen.models.attentions import SoftmaxAttention, LinearAttention, ...` keeps working) but deletes `ATTENTION_CLASSES` and `build_attention`, along with the now-unused `typing`/`torch` imports and the registry-oriented docstring. Re-exporting classes is not a name→class mapping and does not reintroduce selection by string.

- Alternative considered: remove the re-exports and require submodule imports (`from compgen.models.attentions.softmax import SoftmaxAttention`). Rejected as churn with no behavioral benefit.

### 3. Task models stop encoding architecture in `self.name`
`self.name` currently embeds `attention_type`, which the model will no longer know. Set it to a configuration-only label (`match3_h{hidden_dim}_l{num_layers}_heads{num_heads}`, likewise for fuzzy-logic). `self.name` is not read anywhere in the repository; run naming uses `Variant.name`, which keeps its label.

### 4. The runner keeps its string labels and resolves them locally
`ExperimentConfig.attention_types` stays `Tuple[str, ...]` (default `("softmax", "linear")`), so the CLI (`--attention-types`), `scripts/sweep_fuzzy_logic.py` (`replace`/`asdict`/JSON hashing), and `scripts/grids/*.json` are unchanged. `Variant.attention_type` stays the label for run naming and metrics.

The label→class resolution and object construction move to the runner: `run.py` builds a zero-arg factory from the variant's label and the resolved model/device configuration, and passes it to the model. The runner holds a small explicit table of its two supported architectures (`softmax`, `linear`) using the concrete classes. This is an experiment-local detail, not a model-layer registry; the old model-layer requirement forbidding task-local name mappings is removed by this change.

- Why not store classes/factories directly in the config: the sweep grid and CLI are string/JSON based and hash their override values; factories are not JSON-serializable, and the device is only resolved at run time (inside `run_experiment`), after model configuration.
- Why not a dotted-path CLI: it changes the CLI/grid surface for no benefit at this scale.

### 5. No compatibility layer
The old names are removed outright; there is no deprecation alias. Callers that used `attention_type=` or `build_attention`/`ATTENTION_CLASSES` must switch to constructing attention and passing a factory.

### 6. Documentation and notebooks updated in the same change
`AGENTS.md` and the root `README.md` lose the registry/builder wording and describe direct construction. The two Match3 notebooks that pass `attention_type=` are updated to build factories; every affected construction site is found by grepping for `attention_type`, `build_attention`, and `ATTENTION_CLASSES`.

## Risks / Trade-offs

- **Hidden callers of the old surface** (notebooks, scripts, external code) break at construction. Mitigation: grep the whole repo for `attention_type`, `build_attention`, `ATTENTION_CLASSES` and update every hit; the change is intentionally breaking and ships with the doc/notebook updates.
- **Runner-local label table can drift** from the concrete classes. Mitigation: it is two entries and lives next to the only caller that needs it; the model layer has no table to drift against.
- **Behavioral drift in the runner** (wrong architecture constructed per variant). Mitigation: keep `Variant`/run naming and metrics keyed by the same label, and verify the four-variant default still produces softmax and linear at zero and positive coefficient.
- **Re-exporting classes from `attentions/__init__.py` could invite a new registry.** Mitigation: docstring states explicitly there is no registry or builder and callers construct classes directly.

## Migration Plan

1. Remove `ATTENTION_CLASSES`/`build_attention` and update the package docstring.
2. Switch both task models to `attention_factory`; adjust `self.name`.
3. Update the runner (`run.py` factory construction; `config.py` drops the `ATTENTION_CLASSES` import and built-in name filter) while keeping `attention_types` labels and CLI/grid behavior.
4. Update the Match3 notebooks and both docs.
5. Rollback: the change is a normal commit; reverting restores the registry and the `attention_type` surface.
