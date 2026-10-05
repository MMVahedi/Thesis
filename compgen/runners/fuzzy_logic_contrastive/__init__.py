"""Fuzzy-logic four-model contrastive experiment.

Composes the fuzzy-logic task model, the on-the-fly fuzzy-logic dataset, the
shared-term contrastive loss, the training engine, and experiment tracking to
compare softmax and linear attention, each with and without the proposed
response-token attention-code contrastive term.

Run it with::

    python -m compgen.runners.fuzzy_logic_contrastive

See ``config.py`` for the configuration, ``step.py`` for the training step,
and ``run.py`` for the orchestration.
"""