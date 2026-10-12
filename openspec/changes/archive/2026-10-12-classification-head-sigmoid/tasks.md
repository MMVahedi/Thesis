# Tasks

## 1. Classification head applies sigmoid

- [x] 1.1 In `compgen/models/heads.py`, apply `torch.sigmoid` as the final step of `TokenClassifier.forward` and update its docstring to state it returns probabilities in `(0, 1)` and that training should use a probability-input BCE loss. Verify from the repo root with `venv/bin/python` that a `TokenClassifier(hidden_dim=8)` forward pass on a `(2, 5, 8)` float64 tensor returns shape `(2, 5, 1)` with every value strictly between 0 and 1.
- [x] 1.2 Verify `RegressionHead` is unaffected: construct it and confirm its forward pass on the same input returns `(2, 5, 1)` values that are not constrained to `(0, 1)` (a sigmoid must not have leaked into the regression path).

## 2. Match3 consumes the head's probabilities

- [x] 2.1 In `compgen/models/tasks/match3.py`, remove the `torch.sigmoid(logits)` step and assign the classifier output directly to `y_hat`, keeping the returned tuple shape and `return_attention` behavior unchanged. Verify from the repo root that a `Match3Model` built with a softmax attention factory returns per-token predictions strictly in `(0, 1)` that equal `model.classifier(final_hidden)` with no additional normalization, and that `return_attention=True` still returns three elements.
- [x] 2.2 Confirm nothing else depends on a logit output: grep `compgen/` and `notebooks/` for `TokenClassifier`, `BCEWithLogitsLoss`, and `sigmoid`, and verify the only consumer is `Match3Model` and the notebooks' `nn.BCELoss()` path still matches the model's probability output.

## 3. Spec coherence and validation

- [x] 3.1 Run `openspec validate classification-head-sigmoid --strict` and resolve any reported issue so the change validates.
