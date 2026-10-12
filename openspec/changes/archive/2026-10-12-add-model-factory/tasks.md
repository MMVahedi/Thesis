# Tasks

## 1. Configuration types and selection enums

- [x] 1.1 Add `AttentionArchitecture` and `FeedForwardVariant` enums in `compgen/models/configs.py`, covering only the sequence-composable attention architectures (softmax, linear, strassen, third_order; omit triangular) and both feed-forward variants (relu stack, gelu). Verify with `python -c "from compgen.models.configs import AttentionArchitecture, FeedForwardVariant; print(list(AttentionArchitecture), list(FeedForwardVariant))"`.
- [x] 1.2 Add a shared config base plus `Match3Config` and `FuzzyLogicConfig` dataclasses exposing the variation knobs (attention enum, `num_heads`, `qk_head_dim`, `v_head_dim`, `hidden_dim`, `num_layers`, feed-forward enum with `ffn_dim`/`ffn_depth`, `dropout_rate`, `use_norm`, `share_layers` where supported, task-specific inputs such as `input_dim`/`embedding_norm_scalar`, and a `WeightInitializer` subclass initializer). Verify each config constructs with its defaults and that selecting `Match3Config` vs `FuzzyLogicConfig` is distinguishable by type.
- [x] 1.3 Ensure no raw-string selector is accepted: a config or factory call supplying a plain string for attention/feed-forward selection must fail rather than resolve. Verify by attempting a string selection in a `python -c` check and confirming it raises.

## 2. Task models accept a constructed feed-forward module

- [x] 2.1 Change `Match3Model` to take `ffn: nn.Module` (replacing `ffn_depth`) and use that module as the feed-forward of every encoder layer, mirroring the existing `attention: nn.Module`; keep initialization and `share_layers` behavior. Verify by constructing `Match3Model` with explicit attention/ffn modules and running a forward pass on a sample batch, checking output shapes match the previous contract.
- [x] 2.2 Change `FuzzyLogicModel` to take `ffn: nn.Module` (replacing `ffn_dim`) and use that module as the feed-forward of every encoder layer, mirroring the existing `attention: nn.Module`. Verify by constructing `FuzzyLogicModel` with explicit attention/ffn modules and running a forward pass, checking prediction shape and one attention matrix per layer.
- [x] 2.3 Confirm both task models reject the removed parameters (`ffn_depth`/`ffn_dim`, and any architecture-name/variant string) and accept only constructed modules. Verify with a `python -c` check asserting the old keyword raises `TypeError`.

## 3. Model factory

- [x] 3.1 Add enum→concrete-class mappings and `build_model(config)` in `compgen/models/factory.py`, dispatching per task with explicit `isinstance` branches and constructing the attention module, feed-forward module, embedding, head, and initialization. Verify `build_model(Match3Config(...))` and `build_model(FuzzyLogicConfig(...))` each return an `nn.Module` that runs a forward pass without a caller-supplied sub-module.
- [x] 3.2 Verify every knob is honored: build two models differing in one knob at a time (attention architecture, head count/dims, feed-forward variant, width/depth, layers/width, dropout, norm, share_layers, initializer) and confirm the resulting models reflect each value (e.g. encoder layer's attention class, feed-forward class, stack length).
- [x] 3.3 Verify factory/direct equivalence: build a model through the factory and construct the equivalent architecture directly with the same arguments, then confirm they expose the same architecture configuration (stack depth, widths, head count, attention architecture, feed-forward variant).
- [x] 3.4 Re-export the factory and configs from `compgen/models/__init__.py` (and `compgen/models/configs.py` exports) if the package exposes public names, and verify `from compgen.models import ...` resolves them.

## 4. Documentation

- [x] 4.1 Update `AGENTS.md` and the root `README.md` to describe the model-layer factory, enum-based selection (no raw strings), the per-task configs, and the revised add-a-task convention (embedding + thin task file + config/factory branch). Verify every model-layer path mentioned in both docs exists on disk.
- [x] 4.2 Search `AGENTS.md` and `README.md` for stale wording (name-to-architecture registry, attention construction helper, `attention_factory`) and confirm the updated text no longer contradicts reality.

## 5. Integration check

- [x] 5.1 End-to-end smoke check: build a fuzzy-logic model and a Match3 model through the factory from configs, instantiate the relevant dataset/collate (or a synthetic batch), and confirm a forward pass runs for each produced model without direct sub-module construction.
