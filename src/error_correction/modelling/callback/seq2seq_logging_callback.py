#
# Copyright (c) 2019-2021 James Thorne.
#
# This file is part of factual error correction.
# See https://jamesthorne.co.uk for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from pathlib import Path
import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from error_correction.modelling.callback.checkpoint_callback import (
    count_trainable_parameters,
)

logger = logging.getLogger(__name__)


class Seq2SeqLoggingCallback(pl.Callback):
    @rank_zero_only
    def _write_logs(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        type_path: str,
        save_generations=True,
    ) -> None:
        logger.info(
            f"***** {type_path} results at step {trainer.global_step:05d} *****"
        )
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics(
            {
                k: v
                for k, v in metrics.items()
                if k not in ["log", "progress_bar", "preds"]
            }
        )
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = (
                od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            )
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics(
            {"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6}
        )

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        return self._write_logs(trainer, pl_module, "test")
