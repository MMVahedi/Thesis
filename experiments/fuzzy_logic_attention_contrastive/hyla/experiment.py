"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
import csv
import enum
import json
import os
import pickle
import time
from functools import partial
from typing import Callable, Dict, Tuple, Type

import chex
import flax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import struct
from tqdm.auto import tqdm

from hyla.data.base import Dataset, Dataloader
from hyla.logging import Logger, StandardLogger

Metrics = Dict[str, chex.Array]


class CallbackEvent(enum.Enum):
    START = enum.auto()
    STEP = enum.auto()
    END = enum.auto()


class Callback(abc.ABC):
    """
    Callbacks are expected to take care of jit-compiling themselves if possible.
    """
    def __init__(self, log_level: int, onevent: CallbackEvent) -> None:
        self.log_level = log_level
        self.onevent = onevent

    @abc.abstractmethod
    def __call__(self, ctx, exp_state) -> Metrics:
        pass


@struct.dataclass
class ExperimentLoss(abc.ABC):
    apply_fn: Callable

    @abc.abstractmethod
    def __call__(
        self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool
    ) -> Tuple[float, Metrics]:
        pass


@struct.dataclass
class ExperimentState:
    optim: optax.OptState
    params: Dict
    rng: chex.PRNGKey
    step: int


class Experiment:
    def __init__(
        self,
        config: Dict,
        model: flax.linen.Module,
        loss: Type[ExperimentLoss],
        optimizer: optax.GradientTransformation,
        train_loader: Dataloader,
        eval_loaders: Dict[str, Dataloader],
        callback_loaders: Dict[str, Dataloader],
        logger: Tuple[Logger] = [StandardLogger()],
        callbacks: Tuple[Callback] = tuple(),
        log_level: int = 0,
    ):
        self.config = config
        self.model = model
        self.loss_fn = loss(apply_fn=self.model.apply)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders
        self.callback_loaders = callback_loaders
        self.logger = logger
        self.callbacks = callbacks
        self.log_level = log_level

    def trigger_callback(self, exp_state: ExperimentState, onevent: CallbackEvent):
        metrics = dict()
        for c in self.callbacks:
            if c.onevent == onevent and c.log_level <= self.log_level:
                metrics.update(c(exp_state=exp_state, ctx=self))

        return metrics

    def log(self, step: int, log_dict: Dict, prefix: str = ""):
        if not log_dict:
            return
        for l in self.logger:
            l.log(step, {prefix + "_" + k: log_dict[k] for k in log_dict})

    def reset(self, rng: chex.PRNGKey) -> ExperimentState:
        rng_exp, rng_params, rng_dropout = jax.random.split(rng, 3)
        sample_batch = next(iter(self.train_loader))
        init_rngs = {'params': rng_params, 'dropout': rng_dropout}
        params = self.model.init(init_rngs, sample_batch.x, deterministic=True)
        optim = self.optimizer.init(params)

        return ExperimentState(optim=optim, params=params, rng=rng_exp, step=0)

    @staticmethod
    def load(directory: str) -> Tuple[ExperimentState, Dict]:
        config = pickle.load(open(os.path.join(directory, "config.pkl"), "rb"))
        state = pickle.load(open(os.path.join(directory, "state.pkl"), "rb"))
        return config, state

    def save(self, exp_state: ExperimentState):
        pickle.dump(self.config, open(os.path.join(self.config.log_dir, "config.pkl"), "wb"))
        pickle.dump(exp_state, open(os.path.join(self.config.log_dir, "state.pkl"), "wb"))

    def run(self, exp_state: ExperimentState):
        evaluation_history = []
        latest_eval = {}
        samples_per_epoch = len(self.train_loader) * self.config.batch_size
        num_epochs = int(self.config.num_epochs)
        if num_epochs < 1:
            raise ValueError("config.num_epochs must be at least 1")

        # Trigger callbacks on CallbackEvent.START
        self.log(exp_state.step, self.trigger_callback(
            exp_state, CallbackEvent.START), "callback")

        def evaluate_epoch(state, epoch, samples_seen):
            checkpoint = {}
            for name, eval_loader in self.eval_loaders.items():
                eval_metrics = self.eval(state, eval_loader)
                self.log(state.step, eval_metrics, prefix=name)
                metrics_float = {key: float(value) for key, value in eval_metrics.items()}
                evaluation_history.append({
                    "epoch": epoch,
                    "step": int(state.step),
                    "samples_seen": samples_seen,
                    "split": name,
                    **metrics_float,
                })
                checkpoint[f"{name}_mse"] = metrics_float["task_loss"]
                checkpoint[f"{name}_r2"] = metrics_float["r2"]
                checkpoint[f"{name}_avg_num_positives"] = metrics_float[
                    "avg_num_positives"
                ]

            history_path = os.path.join(self.config.log_dir, "evaluation_history.csv")
            with open(history_path, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=evaluation_history[0].keys())
                writer.writeheader()
                writer.writerows(evaluation_history)

            return checkpoint

        samples_seen = 0
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            epoch_metric_sums = None
            epoch_steps = 0
            progress = tqdm(
                total=samples_per_epoch,
                desc=(
                    f"epoch {epoch}/{num_epochs} | "
                    f"lambda={self.config.lambda_contrastive:g}"
                ),
                unit="sample",
                mininterval=1.0,
                dynamic_ncols=True,
            )
            for batch in iter(self.train_loader):
                exp_state, metrics = self.train_step(exp_state, batch)
                if epoch_metric_sums is None:
                    epoch_metric_sums = metrics
                else:
                    epoch_metric_sums = jtu.tree_map(
                        lambda total, value: total + value,
                        epoch_metric_sums,
                        metrics,
                    )
                epoch_steps += 1
                batch_samples = int(batch.x.shape[0])
                samples_seen += batch_samples
                progress.update(batch_samples)
            progress.close()

            train_metrics = jtu.tree_map(
                lambda total: total / epoch_steps,
                epoch_metric_sums,
            )
            elapsed = max(time.time() - epoch_start, 1e-12)
            train_metrics = {
                **train_metrics,
                "samples_per_sec": samples_per_epoch / elapsed,
            }
            self.log(exp_state.step, train_metrics, prefix="train")

            latest_eval = evaluate_epoch(exp_state, epoch, samples_seen)
            tqdm.write(
                f"epoch {epoch}/{num_epochs} complete | "
                f"samples={samples_seen:,} | "
                + " | ".join(
                    f"{key}={value:.6f}" for key, value in latest_eval.items()
                )
            )
            self.log(exp_state.step, self.trigger_callback(
                exp_state, CallbackEvent.STEP), "callback")

        final_result = {
            "task_name": self.config.name.split(";")[0],
            "num_variables": int(self.config.data.num_variables),
            "num_terms": int(self.config.data.num_terms),
            "training_steps": int(exp_state.step),
            "num_epochs": num_epochs,
            "dataset_size": int(self.config.data.num_train),
            "samples_seen": samples_seen,
            "batch_size": int(self.config.batch_size),
            "seq_len": int(self.config.data.seq_len),
            "seed": int(self.config.seed),
            "lambda_contrastive": float(self.config.lambda_contrastive),
            "temperature": float(self.config.temperature),
            "learning_rate": float(self.config.lr),
            "weight_decay": float(self.config.weight_decay),
            "attention_dropout_rate": float(self.config.model.attention_dropout_rate),
            **latest_eval,
        }
        final_metrics_path = os.path.join(self.config.log_dir, "final_metrics.json")
        temporary_metrics_path = final_metrics_path + ".tmp"
        with open(temporary_metrics_path, "w") as stream:
            json.dump(final_result, stream, indent=2, sort_keys=True)
        os.replace(temporary_metrics_path, final_metrics_path)

        tqdm.write(
            f"completed {num_epochs} epoch(s), {samples_seen:,} samples, "
            f"{int(exp_state.step):,} optimizer steps | "
            + " | ".join(f"{key}={value:.6f}" for key, value in latest_eval.items())
        )

        # Trigger callbacks on CallbackEvent.END
        self.log(exp_state.step, self.trigger_callback(exp_state, CallbackEvent.END), "callback")

        return exp_state

    def eval(self, exp_state: ExperimentState, eval_loader: Dataloader):
        metrics_list, rng = [], exp_state.rng

        for batch in iter(eval_loader):
            rng, rng_test = jax.random.split(rng)
            metrics_list.append(self.eval_step(exp_state, rng_test, batch))

        metrics = jtu.tree_map(lambda *args: jnp.stack((args)), *metrics_list)
        metrics = jtu.tree_map(lambda x: jnp.mean(x, axis=0), metrics)

        return metrics

    @partial(jax.jit, static_argnames="self")
    def eval_step(self, exp_state: ExperimentState, rng: chex.PRNGKey, batch: Dataset) -> Dict:
        return self.loss_fn(exp_state.params, rng, batch, deterministic=True)[1]

    @partial(jax.jit, static_argnames="self")
    def train_step(self, exp_state: ExperimentState, batch: Dataset) -> Tuple[ExperimentState, Dict]:
        rng_grad, rng_new = jax.random.split(exp_state.rng)
        (loss, metrics), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True)(exp_state.params, rng_grad, batch, deterministic=False)

        params_update, optim = self.optimizer.update(grads, exp_state.optim, exp_state.params)
        params = optax.apply_updates(exp_state.params, params_update)

        exp_state = ExperimentState(
            params=params,
            optim=optim,
            rng=rng_new,
            step=exp_state.step + 1,
        )

        metrics.update({
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(params),
        })

        return exp_state, metrics
