"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Dict

import chex
import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import lax

from hyla.data.base import Dataset
from hyla.experiment import ExperimentLoss
from hyla.utils.dict import dict_filter


@struct.dataclass
class SymbolicRavenLoss(ExperimentLoss):
    num_features: int
    num_feature_values: int

    def __call__(self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool):

        kwargs = dict(deterministic=deterministic, rngs={"dropout": rng})
        logits = self.apply_fn(params, batch.x, **kwargs)[:, -self.num_features:]

        # treat as multiple classification problem
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.y)
        loss = jnp.mean(loss)

        metrics = {
            "loss": loss,
            "acc": jnp.mean(jnp.equal(jnp.argmax(logits, axis=-1), batch.y).all(axis=1)),
            "acc_per_feature": jnp.mean(jnp.equal(jnp.argmax(logits, axis=-1), batch.y)),
        }

        return loss, metrics


@struct.dataclass
class FuzzyLogicLoss(ExperimentLoss):
    lambda_contrastive: float = 0.0
    temperature: float = 1.0
    num_layers: int = 2

    def __call__(self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool):

        kwargs = dict(deterministic=deterministic, rngs={"dropout": rng})
        if self.lambda_contrastive > 0.0:
            logits, variables = self.apply_fn(
                params, batch.x, **kwargs, mutable="intermediates")
            final_attention = dict_filter(
                variables,
                [f"transformer_block_{self.num_layers - 1}", "attn_weights"],
            )[0]
            attention_representation = final_attention[:, :, -1, -1]
        else:
            logits = self.apply_fn(params, batch.x, **kwargs)
            attention_representation = None

        logits = logits[:, -1]

        squared_error = optax.squared_error(predictions=logits, targets=batch.y)
        task_loss = jnp.mean(squared_error)
        r2 = 1 - (squared_error / batch.info["base_mse"])

        term_ids = batch.info["term_ids"]
        shared_term = jnp.any(
            term_ids[:, None, :, None] == term_ids[None, :, None, :], axis=(2, 3))
        positive_mask = shared_term & ~jnp.eye(term_ids.shape[0], dtype=bool)
        num_positives = jnp.sum(positive_mask, axis=1)
        avg_num_positives = jnp.mean(num_positives)

        if attention_representation is not None:
            z = attention_representation / jnp.maximum(
                jnp.linalg.norm(attention_representation, axis=1, keepdims=True), 1e-12)
            similarity = z @ z.T
            logits_contrastive = similarity / self.temperature
            logits_contrastive = jnp.where(
                jnp.eye(term_ids.shape[0], dtype=bool), -jnp.inf, logits_contrastive)
            log_prob = logits_contrastive - jax.nn.logsumexp(
                logits_contrastive, axis=1, keepdims=True)

            valid_anchor = num_positives > 0
            loss_per_anchor = -jnp.sum(
                jnp.where(positive_mask, log_prob, 0.0), axis=1
            ) / jnp.maximum(num_positives, 1)
            contrastive_loss = jnp.sum(
                jnp.where(valid_anchor, loss_per_anchor, 0.0)
            ) / jnp.maximum(jnp.sum(valid_anchor), 1)
        else:
            contrastive_loss = jnp.array(0.0, dtype=task_loss.dtype)

        total_loss = task_loss + self.lambda_contrastive * contrastive_loss

        metrics = {
            "task_loss": task_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": total_loss,
            "avg_num_positives": avg_num_positives,
            "r2": jnp.mean(r2),
        }

        return total_loss, metrics


class AutoregressiveCrossEntropy(ExperimentLoss):
    def __call__(self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool):

        def shift_right(x, axis=1):
            """Shift the input to the right by padding and slicing on axis."""
            pad_widths = [(0, 0)] * len(x.shape)
            pad_widths[axis] = (1, 0)
            padded = jnp.pad(x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
            return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)

        inputs = shift_right(batch.x)
        targets = batch.x
        weights = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)
        denominator = jnp.sum(weights)

        logits = self.apply_fn(
            params, inputs, deterministic=deterministic, rngs={"dropout": rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
        loss = jnp.sum(loss * weights) / denominator

        perplexity = jnp.mean(jnp.exp(jnp.sum(loss * weights, axis=1) / jnp.sum(weights, axis=1)))
        acc = jnp.sum(jnp.equal(jnp.argmax(logits, axis=-1), targets) * weights) / denominator

        metrics = {
            "loss": loss,
            "acc": acc,
            "perplexity": perplexity,
        }

        return loss, metrics
