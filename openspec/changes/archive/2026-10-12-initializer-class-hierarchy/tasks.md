# Tasks

## 1. Initializer class hierarchy

- [x] 1.1 Rewrite `compgen/models/initialization.py`: make `WeightInitializer` an abstract base (its `__call__` runs `module.apply(self._init)` and returns the module; `_init(module)` is an abstract method), add the concrete subclasses `XavierInitializer`, `ReferenceInitializer`, and `DefaultInitializer` whose `_init` bodies are today's `_init_xavier_style`/`_init_reference_style`/`_init_default_style`, delete the three free scheme functions, and update the module docstring to describe the class hierarchy; verify each subclass constructs and applies its expected scheme to a tiny `nn.Sequential`, and that instantiating the abstract base raises `TypeError`.
- [x] 1.2 Delete the module-level instances `XAVIER_INIT`, `REFERENCE_INIT`, and `DEFAULT_INIT`, remove any string `name`/scheme identifier (make `__repr__` use the concrete class name), and keep `init_reference_weights` as a thin wrapper over `ReferenceInitializer`; verify `init_reference_weights` and `ReferenceInitializer` produce identical weights on identical modules and that no string-keyed lookup or string type field remains.

## 2. Attention architectures

- [x] 2.1 In `compgen/models/attentions/base.py`, change the constructor parameter to `initializer: Type[WeightInitializer] = XavierInitializer`, instantiate it in `__init__` before `_init_weights` runs, keep `_init_weights` delegating to the instance, and update the module/class docstrings; verify constructing the base through a concrete architecture yields xavier-normal weights with zero biases with no external call.
- [x] 2.2 Update the five architecture modules (`softmax`, `linear`, `strassen`, `third_order`, `triangular`) to import the classes instead of the singleton, take `initializer: Type[WeightInitializer] = XavierInitializer`, and forward it to `super().__init__`; verify each constructs with the xavier default and, when constructed with `ReferenceInitializer`, its projection parameters follow the reference scheme instead.

## 3. Prediction heads

- [x] 3.1 In `compgen/models/heads.py`, change `TokenClassifier` to default to `XavierInitializer` (importing the class), instantiate and apply it, and update its docstring; verify a constructed classifier's weights follow xavier-normal with zero biases and that a supplied `ReferenceInitializer` changes the scheme.
- [x] 3.2 Change `RegressionHead` to default to `DefaultInitializer` (importing the class) and update its docstring; verify a default-constructed head keeps PyTorch defaults with its `LayerNorm` untouched, while a supplied `XavierInitializer` changes the linear weights.

## 4. Feed-forward variants

- [x] 4.1 In `compgen/models/ffn.py`, change `BaseFeedForward` to take `initializer: Type[WeightInitializer] = DefaultInitializer`, instantiate it, and have `_init_weights` apply it; update the module docstring; verify both variants construct and forward at unchanged width and that default-constructed weights equal the previous PyTorch defaults.
- [x] 4.2 Verify injection: construct each variant with `XavierInitializer` and confirm its `nn.Linear` weights follow xavier-normal with zero biases.

## 5. Task embeddings

- [x] 5.1 In `compgen/models/embeddings/fuzzy_logic.py` and `compgen/models/embeddings/match3.py`, change the parameter to `initializer: Type[WeightInitializer] = DefaultInitializer`, instantiate and apply it after the projection is built (guarding Match3's absent `in_proj`), and update the docstrings; verify both embeddings construct and forward with unchanged output shapes and that default weights equal PyTorch defaults.
- [x] 5.2 Verify injection: construct each embedding with `XavierInitializer` and confirm its projection weights follow xavier-normal.

## 6. Encoder layer

- [x] 6.1 In `compgen/models/encoder.py`, change the parameter to `initializer: Type[WeightInitializer] = DefaultInitializer`, instantiate it, and apply it only to the layer's own normalization parameters; update the docstring to note the layer still never initializes or re-initializes the modules it is handed; verify passing an attention module with known weights leaves its `state_dict()` unchanged and that `use_norm=False` yields no LayerNorm or own parameters.

## 7. Task models

- [x] 7.1 In `compgen/models/tasks/fuzzy_logic.py`, change the parameter to `initializer: Type[WeightInitializer] = ReferenceInitializer`, instantiate it in `__init__`, apply it across the assembled model, and update the docstring/imports; verify a default-constructed `FuzzyLogicModel` follows the reference scheme across its modules (attention included) and that parameter count and forward output shapes are unchanged.
- [x] 7.2 In `compgen/models/tasks/match3.py`, change the parameter to `initializer: Type[WeightInitializer] = DefaultInitializer`, instantiate and apply it, and update the docstring; verify a default-constructed `Match3Model` keeps the previous per-module schemes and that parameter count and forward output shapes are unchanged.
- [x] 7.3 Verify a non-default policy overrides the composition: construct each task model with `XavierInitializer` and confirm the whole model's linear weights follow that scheme.

## 8. Documentation

- [x] 8.1 Update the model-layer description in `AGENTS.md` and the root `README.md` to describe the initializer class hierarchy (one class per scheme, no string type) and that modules receive the policy class; verify every repository-relative model-layer path mentioned still exists and no text references the removed `XAVIER_INIT`/`REFERENCE_INIT`/`DEFAULT_INIT` or a string scheme name.

## 9. Integration verification

- [x] 9.1 From the repo root with a fixed seed, build one `Match3Model` and one `FuzzyLogicModel` and confirm total parameter counts, forward output shapes, and fixed-seed state-dict checksums match the pre-change revision exactly.
- [x] 9.2 Build a model through `compgen/runners/fuzzy_logic_contrastive/run.py:build_model` (which passes no initializer) and confirm it constructs and forwards successfully.
- [x] 9.3 Verify removal is complete: search `compgen/` for `XAVIER_INIT`, `REFERENCE_INIT`, and `DEFAULT_INIT` and confirm no references remain, and that `compgen/models/initialization.py` exposes no string-keyed scheme selection.
