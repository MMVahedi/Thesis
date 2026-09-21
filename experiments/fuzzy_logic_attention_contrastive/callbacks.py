"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import pickle

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from einops import rearrange
from ml_collections import ConfigDict

from hyla.experiment import Callback, CallbackEvent, Metrics
from hyla.utils.dict import dict_filter


class RavenLatents(Callback):
    def __init__(
            self,
            log_level: int,
            onevent: CallbackEvent,
            data_config: ConfigDict,
    ) -> None:
        super().__init__(log_level, onevent)
        self.data_config = data_config

    def __call__(self, ctx, exp_state) -> Metrics:
        @jax.jit
        def predict(batch):
            logits, variables = ctx.model.apply(
                exp_state.params, batch.x, deterministic=True, mutable="intermediates")

            # Extract attention weights for the last feature (wlog)
            attn_weights = dict_filter(variables, "attn_weights")
            attn_weights = jnp.stack(attn_weights, axis=1)  # shape=(batch, layer, head, query, key)

            g, f = self.data_config.grid_size, self.data_config.num_features
            grid_indeces = rearrange(np.arange(g**2 * f), "(k n m)-> k n m", k=g, n=g, m=f)
            input_indeces = rearrange(grid_indeces[:, :, -1], "k n -> (k n)")[-g:-1]
            input = jnp.argmax(batch.x[:, input_indeces], axis=-1)
            attn_weights = jnp.take_along_axis(
                attn_weights, rearrange(input_indeces, "n -> 1 1 1 1 n"), axis=-1)

            # extract data for last raven-feature only (wlog)
            return {
                "attn_weight": attn_weights[:, :, :, -1:, :],     # shape=(batch, layer, head, query=-1, key=grid_size - 1)
                "latent": batch.info["latents"][:, -1],           # shape=(batch,)
                "input": input,                                   # shape=(batch, grid_size - 1)
                "target": batch.y[:, -1],                         # shape=(batch,)
                "prediction": jnp.argmax(logits[:, -1], axis=-1)  # shape=(batch,)
            }

        loaders = {
            "test_unpermuted": ctx.callback_loaders["test_unpermuted"],
            "ood_unpermuted": ctx.callback_loaders["ood_unpermuted"],
        }

        datasets = {}
        for name, loader in loaders.items():
            data_list = []
            for batch in loader:
                data_list.append(predict(batch))

            datasets[name] = jtu.tree_map(lambda *args: jnp.concatenate((args)), *data_list)

        # Save to disk into config.log_dir
        datasets_numpy = jtu.tree_map(lambda x: np.array(x), datasets)
        pickle.dump(datasets_numpy, open(os.path.join(
            ctx.config.log_dir, f"raven_latents_step_{exp_state.step}.pkl"), "wb"))

        return dict()


class LogicLatents(Callback):
    """
    Callbacks are expected to take care of jit-compiling themselves if possible.
    """
    def __init__(self, log_level: int, onevent: CallbackEvent, save_to_disk: bool) -> None:
        super().__init__(log_level, onevent)
        self.save_to_disk = save_to_disk

    def __call__(self, ctx, exp_state) -> Metrics:
        @jax.jit
        def predict(batch):
            _, variables = ctx.model.apply(
                exp_state.params, batch.x, deterministic=True, mutable="intermediates")

            # Extract attention weights for the last sample in-context
            attn_weights = dict_filter(variables, "attn_weights")
            attn_weights = jnp.stack([  # shape: (batch, layer, head, query, key)
                a[:, :, -1, -1]
                for a in attn_weights
            ], axis=1)

            return attn_weights, batch.info["latents"]

        # The paper analyzes response-token attention codes while keeping the
        # in-context inputs fixed, then tests term decoding on held-out tasks.
        loaders = {
            "train_fixed_context": ctx.callback_loaders["valid_fixed_context"],
            "test_fixed_context": ctx.callback_loaders["test_fixed_context"],
        }
        datasets = {}
        for name, loader in loaders.items():
            data_list = []
            for batch in loader:
                attn, latent = predict(batch)
                data_list.append({"attn_weight": attn, "latent": latent, "target": batch.y})

            datasets[name] = jtu.tree_map(lambda *args: jnp.concatenate((args)), *data_list)

        # Save to disk into config.log_dir
        if self.save_to_disk:
            datasets_numpy = jtu.tree_map(lambda x: np.array(x), datasets)
            pickle.dump(datasets_numpy, open(os.path.join(
                ctx.config.log_dir, f"latent_dataset_step_{exp_state.step}.pkl"), "wb"))

        return dict()
