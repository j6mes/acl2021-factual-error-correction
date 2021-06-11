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
import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl

from error_correction.modelling.callback.logging_callback \
    import LoggingCallback
from error_correction.modelling.base_transformer import BaseTransformer

logger = logging.getLogger(__name__)


def add_generic_args(parser, root_dir) -> None:
    #  TODO(SS): allow all pl args?
    #   parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model "
             "predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) "
             "precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level "
             "selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--n_tpu_cores",
                        dest="tpu_cores", type=int, default=None)
    parser.add_argument(
        "--max_grad_norm",
        dest="gradient_clip_val",
        default=1.0,
        type=float,
        help="Max gradient norm",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate "
             "before performing a backward/update pass.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )


def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    early_stopping_callback=False,
    logger=True,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    pl.seed_everything(args.seed)

    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir,
            prefix="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)

    return trainer
