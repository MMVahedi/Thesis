"""Runnable experiment entry points that compose the ``compgen`` package.

Unlike the isolated archives under ``compgen/experiments/`` (which must not
import the enclosing package), runners here import and compose the package's
model, dataset, loss, training, and tracking layers to run reproducible studies.
"""