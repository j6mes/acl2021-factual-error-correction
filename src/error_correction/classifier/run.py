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
import glob
import logging
import os
from pathlib import Path
import pytorch_lightning as pl

from error_correction.modelling.callback.checkpoint_callback import (
    get_checkpoint_callback,
)
from error_correction.modelling.callback.seq2seq_logging_callback import (
    Seq2SeqLoggingCallback,
)
from error_correction.modelling.lightning_base import generic_train
from error_correction.modelling.classifier_module import FEVERClassifierModule

logger = logging.getLogger(__name__)


def main(args, model=None) -> FEVERClassifierModule:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )
    if model is None:
        model: FEVERClassifierModule = FEVERClassifierModule(args)

    logger = True

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        logger=logger,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(
        sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True))
    )
    print(checkpoints)
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)
    trainer.model = model
    # test() without a model tests using the best checkpoint automatically
    trainer.test(ckpt_path=checkpoints[-1])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = FEVERClassifierModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
