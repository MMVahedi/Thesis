# Tasks

## 1. Remove initialization from sub-level modules

- [x] 1.1 In `compgen/models/attentions/base.py`, delete the `initializer` constructor parameter, the `self.initializer` attribute, and the `_init_weights` method, remove the `initialization.py` import, and update the module/class docstrings to say the architecture does not initialize itself; verify a concrete architecture constructs without any `initializer` argument and its projection weights are PyTorch's constructor defaults.
- [x] 1.2 In the five architecture modules (`softmax`, `linear`, `strassen`, `third_order`, `triangular`), delete the `initializer` parameter, the `initializer=initializer` argument to `super().__init__`, and the `self._init_weights()` call, and remove the `initialization.py` import; verify each constructs with no `initializer` argument and no longer self-initializes.
- [x] 1.3 In `compgen/models/heads.py`, delete the `initializer` parameter, the `self.initializer` attribute, the self-initialization call, and the `initialization.py` import from both `TokenClassifier` and `RegressionHead`, and update their docstrings; verify both construct with no `initializer` argument and keep PyTorch's default parameters (LayerNorm untouched).
- [x] 1.4 In `compgen/models/ffn.py`, delete the `initializer` parameter, the `self.initializer` attribute, and the `_init_weights` method from `BaseFeedForward`, remove the parameter/argument and the `self._init_weights()` call from `ReluStackFFN` and `GeluFFN`, remove the `initialization.py` import, and update the module docstring; verify both variants construct with no `initializer` argument, keep PyTorch defaults, and forward at unchanged width.
- [x] 1.5 In `compgen/models/embeddings/fuzzy_logic.py` and `compgen/models/embeddings/match3.py`, delete the `initializer` parameter, the `self.initializer` attribute, the self-initialization call, and the `initialization.py` import, and update the docstrings; verify both embeddings construct with no `initializer` argument and forward with unchanged output shapes.
- [x] 1.6 In `compgen/models/encoder.py`, delete the `initializer` parameter, the `self.initializer` attribute, and the two `self.initializer(...)` calls on the normalization modules, remove the `initialization.py` import, and update the module docstring; verify the layer constructs with no `initializer` argument, still never initializes the attention/feed-forward modules it is handed, and with `use_norm=False` holds no LayerNorm or own parameters.
- [x] 1.7 Verify sub-level removal is complete: search the five architecture modules, `attentions/base.py`, `heads.py`, `ffn.py`, both embeddings, and `encoder.py` for `initializer`, `WeightInitializer`, and `_init_weights`, and confirm none remains.

## 2. Task-model-level initialization

- [x] 2.1 In `compgen/models/tasks/fuzzy_logic.py`, keep `initializer: Type[WeightInitializer] = ReferenceInitializer`, add a single `_init_weights` method that applies the policy to the whole model (`self.initializer(self)`), and call that method at the end of `__init__`; verify the model constructs, that it exposes exactly one initialization method, and that a default-constructed model's Linear/Embedding parameters follow the reference scheme across the whole model.
- [x] 2.2 In `compgen/models/tasks/match3.py`, change the default to `initializer: Type[WeightInitializer] = XavierInitializer`, add the same single `_init_weights` method applying `self.initializer(self)`, and call it at the end of `__init__`; verify a default-constructed model's Linear/Embedding parameters follow xavier across the whole model (attention, classifier, feed-forward, and embedding), and update the docstring.
- [x] 2.3 Verify the model-level override: construct each task model with a supplied non-default policy class (e.g. `DefaultInitializer` for fuzzy-logic, `ReferenceInitializer` for Match3) and confirm every Linear/Embedding parameter follows that scheme, and that the model exposes no per-sub-module initializer.

## 3. Initialization module docs

- [x] 3.1 Update the `compgen/models/initialization.py` module docstring (and `init_reference_weights` if needed) to state that the class hierarchy is the model-level policy set and that a task model applies one policy to its whole model, rather than each module initializing itself; verify the docstring no longer claims every parameterized module self-initializes.

## 4. Documentation

- [x] 4.1 Update the model-layer description in `AGENTS.md` and the root `README.md` to say initialization is applied only by the task model (one method, one uniform policy class) and that sub-level modules do not initialize; verify every repository-relative model-layer path mentioned still exists and no text says sub-level modules self-initialize.

## 5. Integration verification

- [x] 5.1 From the repo root with a fixed seed, build one `Match3Model` and one `FuzzyLogicModel` and confirm total parameter counts and forward output shapes match the pre-change revision. Record that **both** models' fixed-seed state-dict checksums change: removing the sub-level self-init draws shifts the RNG position at model-level apply, so the model's single initialization method draws differently even where the scheme is unchanged (fuzzy-logic stays reference-everywhere; Match3 becomes xavier-everywhere). Confirm a sub-level module built standalone keeps PyTorch defaults.
- [x] 5.2 Build a model through `compgen/runners/fuzzy_logic_contrastive/run.py:build_model` (whose attention factory passes no initializer) and confirm it constructs and forwards successfully.
- [x] 5.3 Verify no sub-level module references the initializer policy: search `compgen/models/attentions`, `compgen/models/heads.py`, `compgen/models/ffn.py`, `compgen/models/embeddings`, and `compgen/models/encoder.py` for `WeightInitializer` and confirm only `compgen/models/initialization.py` and the two task models reference it.
